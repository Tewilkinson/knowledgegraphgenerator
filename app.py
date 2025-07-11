import streamlit as st
import requests
import networkx as nx
import community as community_louvain   # pip install python-louvain
from pyvis.network import Network
import streamlit.components.v1 as components

# 1. LOOKUP: label → Wikidata QID
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": label
    }
    return requests.get(url, params=params).json()["search"][0]["id"]

# 2. TAXONOMY: direct subclasses (“fan-out”) via P279
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
    rows = requests.get(sparql_url, params={"query": query}, headers=headers) \
                   .json()["results"]["bindings"]
    return [
        (r["child"]["value"].rsplit("/", 1)[-1], r["childLabel"]["value"])
        for r in rows
    ]

# 3. SEMANTIC: related neighbours from ConceptNet
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    url = f"http://api.conceptnet.io/related/c/en/{uri}?filter=/c/en&limit={limit}"
    data = requests.get(url).json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0))
        for e in data
    ]

# 4. BUILD the hybrid graph
def build_graph(seed_label, depth, tax_limit, sem_limit):
    seed_q = lookup_qid(seed_label)
    G = nx.Graph()
    G.add_node(seed_q, label=seed_label, type="seed")

    # 1-hop taxonomy
    for q, lbl in get_subclasses(seed_q, tax_limit):
        G.add_node(q, label=lbl, type="taxonomy")
        G.add_edge(seed_q, q, type="taxonomy")

    # 1-hop semantic
    for lbl, w in get_conceptnet_neighbors(seed_label, sem_limit):
        node_id = f"CN:{lbl}"
        G.add_node(node_id, label=lbl, type="semantic")
        G.add_edge(seed_q, node_id, type="semantic", weight=w)

    # optional 2-hop taxonomy
    if depth > 1:
        for q,d in G.nodes(data=True):
            if d["type"]=="taxonomy":
                for q2,lbl2 in get_subclasses(q, tax_limit//2):
                    if not G.has_node(q2):
                        G.add_node(q2, label=lbl2, type="taxonomy")
                    G.add_edge(q, q2, type="taxonomy")

    return G

# 5. Streamlit UI
st.set_page_config(layout="wide")
st.title("🔗 Hybrid Knowledge-Graph Content Clusters")
st.markdown("""
Enter a **seed topic**, pull both  
• **fan-out** (Wikidata subclasses)  
• **related** (ConceptNet neighbours)  
…then build and **auto-cluster** the graph.
""")
seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Taxonomy depth", 1, 2, 1)
tax_lim = st.slider("Subclasses to fetch",     5, 50, 20)
sem_lim = st.slider("Related neighbours to fetch", 5, 50, 20)

if st.button("Generate Graph"):
    with st.spinner("Building & clustering…"):
        G = build_graph(seed, depth, tax_lim, sem_lim)
        # detect communities
        partition = community_louvain.best_partition(G)

    # 6. Render with PyVis
    net = Network(height="650px", width="100%", notebook=False)
    colors = {"seed":"#ff6666","taxonomy":"#66b2ff","semantic":"#aaff66"}

    # add nodes with group=community_id
    for n, d in G.nodes(data=True):
        grp = partition.get(n, 0)
        net.add_node(
            n,
            label=d["label"],
            color=colors[d["type"]],
            title=f"{d['type']} — cluster {grp}",
            group=str(grp)
        )

    # add edges (dashed for semantic)
    for u,v,d in G.edges(data=True):
        if d["type"]=="semantic":
            net.add_edge(u, v, dashes=True)
        else:
            net.add_edge(u, v)

    # **Auto-cluster** each community on load
    for grp in set(partition.values()):
        node_ids = [n for n, cid in partition.items() if cid==grp]
        if len(node_ids)>1:
            net.cluster(nodes=node_ids)

    net.set_options("""
    {
      "nodes": {
        "font": { "size": 14 },
        "scaling": { "label": true }
      },
      "edges": { "smooth": false },
      "physics": { "barnesHut": { "gravitationalConstant": -8000 } }
    }
    """)
    net.save_graph("graph.html")
    components.html(open("graph.html","r").read(), height=660)
