import streamlit as st
import requests
import networkx as nx
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
    data = requests.get(url, params=params).json()
    return data["search"][0]["id"]

# 2. TAXONOMY: Fetch subclasses via P279
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
    results = resp.json()["results"]["bindings"]
    return [
        (r["child"]["value"].rsplit("/", 1)[-1], r["childLabel"]["value"])
        for r in results
    ]

# 3. SEMANTIC: Fetch ConceptNet neighbours
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    url = f"https://api.conceptnet.io/related/c/en/{uri}?filter=/c/en&limit={limit}"
    data = requests.get(url).json()
    out = []
    for entry in data.get("related", []):
        lbl = entry["@id"].split("/")[-1].replace("_", " ")
        w   = entry.get("weight", 0)
        out.append((lbl, w))
    return out

# 4. BUILD GRAPH
def build_graph(seed_label, depth, tax_limit, sem_limit):
    seed_q = lookup_qid(seed_label)
    G = nx.Graph()
    G.add_node(seed_q, label=seed_label, type="seed")

    # fan-out
    for child_q, child_lbl in get_subclasses(seed_q, limit=tax_limit):
        G.add_node(child_q, label=child_lbl, type="taxonomy")
        G.add_edge(seed_q, child_q, type="taxonomy")

    # related
    for nbr_lbl, w in get_conceptnet_neighbors(seed_label, limit=sem_limit):
        node_id = f"CN:{nbr_lbl}"
        G.add_node(node_id, label=nbr_lbl, type="semantic")
        G.add_edge(seed_q, node_id, type="semantic", weight=w)

    # optional 2-hop taxonomy
    if depth > 1:
        for child_q in [n for n,d in G.nodes(data=True) if d["type"]=="taxonomy"]:
            for gc_q, gc_lbl in get_subclasses(child_q, limit=tax_limit//2):
                if not G.has_node(gc_q):
                    G.add_node(gc_q, label=gc_lbl, type="taxonomy")
                G.add_edge(child_q, gc_q, type="taxonomy")
    return G

# 5. STREAMLIT UI
st.title("🔍 Debuggable Knowledge Graph")
seed     = st.text_input("Seed topic", "data warehouse")
depth    = st.slider("Taxonomy depth", 1, 2, 1)
tax_lim  = st.slider("Subclasses (P279)", 5, 50, 20)
sem_lim  = st.slider("Semantic neighbours", 5, 50, 20)

if st.button("Generate Graph"):
    with st.spinner("Building…"):
        G = build_graph(seed, depth, tax_lim, sem_lim)

    # DEBUG: show me what’s inside the graph
    if st.checkbox("⚙️ Show debug info"):
        st.write("▶️ Nodes:", len(G.nodes))
        st.write("▶️ Edges:", len(G.edges))
        st.write("Sample nodes:", list(G.nodes(data=True))[:10])
        st.write("Sample edges:", list(G.edges(data=True))[:10])
    
    # if graph really has nodes, render
    if G.number_of_nodes() == 0:
        st.error("Graph is empty—no nodes to render.")
    else:
        net = Network(height="600px", width="100%", notebook=False)
        color_map = {"seed":"#ff6666", "taxonomy":"#66b2ff", "semantic":"#aaff66"}
        for n, d in G.nodes(data=True):
            net.add_node(n, label=d["label"], color=color_map[d["type"]])
        for u, v, d in G.edges(data=True):
            if d["type"]=="semantic":
                net.add_edge(u, v, dashes=True)
            else:
                net.add_edge(u, v)
        net.set_options("""
        {
          "nodes": { "font": { "size": 14 }, "scaling": { "label": true } },
          "edges": { "smooth": false },
          "physics": { "barnesHut": { "gravitationalConstant": -8000 } }
        }
        """)
        net.save_graph("graph.html")
        components.html(open("graph.html","r").read(), height=620)
