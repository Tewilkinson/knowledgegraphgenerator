import streamlit as st
import requests
import networkx as nx
import community as community_louvain   # pip install python-louvain
from pyvis.network import Network
import streamlit.components.v1 as components

# ——————————————————————————————————————————————————————————————————————————————
# 1. LOOKUP: label → Wikidata QID
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": label
    }
    data = requests.get(url, params=params).json()
    return data["search"][0]["id"]

# ——————————————————————————————————————————————————————————————————————————————
# 2. TAXONOMY: Fetch direct subclasses (“fan-out”) via P279
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def get_subclasses(qid: str, limit: int = 20):
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?child ?childLabel WHERE {{
      ?child wdt:P279 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    headers = {"Accept": "application/sparql-results+json"}
    resp = requests.get(sparql_url, params={"query": query}, headers=headers)
    data = resp.json()["results"]["bindings"]
    return [
        (b["child"]["value"].rsplit("/", 1)[-1], b["childLabel"]["value"])
        for b in data
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 3. SEMANTIC: Fetch neighbours from ConceptNet
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    url = f"http://api.conceptnet.io/related/c/en/{uri}?filter=/c/en&limit={limit}"
    data = requests.get(url).json()
    out = []
    for entry in data.get("related", []):
        lbl    = entry["@id"].split("/")[-1].replace("_", " ")
        weight = entry.get("weight", 0)
        out.append((lbl, weight))
    return out

# ——————————————————————————————————————————————————————————————————————————————
# 4. BUILD A HYBRID GRAPH
# ——————————————————————————————————————————————————————————————————————————————
def build_graph(seed_label, depth, tax_limit, sem_limit):
    seed_q = lookup_qid(seed_label)
    G = nx.Graph()
    G.add_node(seed_q, label=seed_label, type="seed")

    # 1-hop taxonomy (fan-out)
    for child_q, child_lbl in get_subclasses(seed_q, limit=tax_limit):
        G.add_node(child_q, label=child_lbl, type="taxonomy")
        G.add_edge(seed_q, child_q, type="taxonomy")

    # 1-hop semantic (related)
    for nbr_lbl, w in get_conceptnet_neighbors(seed_label, limit=sem_limit):
        nbr_q = f"CN:{nbr_lbl}"
        G.add_node(nbr_q, label=nbr_lbl, type="semantic")
        G.add_edge(seed_q, nbr_q, type="semantic", weight=w)

    # optional 2-hop taxonomy
    if depth > 1:
        for child_q in [n for n,d in G.nodes(data=True) if d["type"]=="taxonomy"]:
            for gc_q, gc_lbl in get_subclasses(child_q, limit=tax_limit//2):
                if not G.has_node(gc_q):
                    G.add_node(gc_q, label=gc_lbl, type="taxonomy")
                G.add_edge(child_q, gc_q, type="taxonomy")

    return G

# ——————————————————————————————————————————————————————————————————————————————
# 5. STREAMLIT UI
# ——————————————————————————————————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("🔗 Hybrid Knowledge-Graph Content Clusters")
st.markdown("""
Enter a **seed topic**, pull both  
– **fan-out** (Wikidata subclasses)  
– **related** (ConceptNet neighbours)  
…then auto-cluster and visualize the graph.
""")

seed     = st.text_input("Seed topic", "data warehouse")
depth    = st.slider("Hierarchy depth (Wikidata)", 1, 2, 1)
tax_lim  = st.slider("Subclasses to fetch", 5, 50, 20)
sem_lim  = st.slider("Related neighbours to fetch", 5, 50, 20)

if st.button("Generate Graph"):
    with st.spinner("Building & clustering…"):
        G = build_graph(seed, depth, tax_lim, sem_lim)

        # detect communities for layout grouping
        partition = community_louvain.best_partition(G)

    # ——————————————————————————————————————————————————————————————————————
    # 6. RENDER WITH PYVIS + CLUSTER BUTTON
    # ——————————————————————————————————————————————————————————————————————
    net = Network(height="650px", width="100%", notebook=False)
    color_map = {"seed":"#ff6666", "taxonomy":"#66b2ff", "semantic":"#aaff66"}

    for n, data in G.nodes(data=True):
        grp = partition.get(n, 0)
        net.add_node(
            n,
            label=data["label"],
            color=color_map[data["type"]],
            title=f"{data['type']} — cluster {grp}",
            group=str(grp)
        )

    for u, v, data in G.edges(data=True):
        if data["type"] == "semantic":
            net.add_edge(u, v, dashes=True)
        else:
            net.add_edge(u, v)

    # cluster-controls UI
    net.show_buttons(filter_=['clustering'])

    # valid JSON options
    net.set_options("""
    {
      "nodes": {
        "font": { "size": 14 },
        "scaling": { "label": true }
      },
      "edges": {
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000
        }
      }
    }
    """)

    net.save_graph("graph.html")
    components.html(open("graph.html", "r").read(), height=660)
