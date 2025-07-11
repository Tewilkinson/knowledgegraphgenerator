import streamlit as st
import requests
import networkx as nx
from streamlit_cytoscapejs import st_cytoscape

# ——————————————————————————————————————————————————————————————————————————————
# 1. LOOKUP: label → Wikidata QID
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label
        },
    )
    resp.raise_for_status()
    return resp.json()["search"][0]["id"]

# ——————————————————————————————————————————————————————————————————————————————
# 2. TAXONOMY: direct subclasses (“fan-out”) via P279
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
    resp.raise_for_status()
    rows = resp.json()["results"]["bindings"]
    return [
        (r["child"]["value"].rsplit("/", 1)[-1], r["childLabel"]["value"])
        for r in rows
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 3. SEMANTIC: related neighbours from ConceptNet
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    url = f"https://api.conceptnet.io/related/c/en/{uri}"
    resp = requests.get(url, params={"filter": "/c/en", "limit": limit})
    resp.raise_for_status()
    entries = resp.json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_", " "), e.get("weight", 0))
        for e in entries
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 4. BUILD the hybrid graph
# ——————————————————————————————————————————————————————————————————————————————
def build_graph(seed: str, depth: int, tax_limit: int, sem_limit: int) -> nx.Graph:
    seed_q = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(seed_q, label=seed, type="seed")

    # 1-hop taxonomy (fan-out)
    for cq, cl in get_subclasses(seed_q, tax_limit):
        G.add_node(cq, label=cl, type="taxonomy")
        G.add_edge(seed_q, cq, type="taxonomy")

    # 1-hop semantic (related)
    for lbl, _ in get_conceptnet_neighbors(seed, sem_limit):
        node_id = f"CN:{lbl}"
        G.add_node(node_id, label=lbl, type="semantic")
        G.add_edge(seed_q, node_id, type="semantic")

    # optional 2-hop taxonomy
    if depth > 1:
        taxonomy_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "taxonomy"]
        for n in taxonomy_nodes:
            for cq, cl in get_subclasses(n, tax_limit // 2):
                if not G.has_node(cq):
                    G.add_node(cq, label=cl, type="taxonomy")
                G.add_edge(n, cq, type="taxonomy")

    return G

# ——————————————————————————————————————————————————————————————————————————————
# 5. STREAMLIT UI
# ——————————————————————————————————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("🔗 Grouped Knowledge-Graph Clusters")
st.markdown(
    """
    Enter a **seed topic**, fetch both  
    • **fan-out** (Wikidata subclasses)  
    • **semantic** neighbours (ConceptNet)  
    …and display them in two grouped containers.
    """
)

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Taxonomy depth (fan-out layers)", 1, 2, 1)
tax_lim = st.slider("Max subclasses (Wikidata P279)", 5, 50, 20)
sem_lim = st.slider("Max related neighbours (ConceptNet)", 5, 50, 20)

if st.button("Generate Graph"):
    G = build_graph(seed, depth, tax_lim, sem_lim)

    # build Cytoscape elements with two containers
    elements = [
        {
            "data": {"id": "fanout", "label": "Fan-out Topics"},
            "classes": "cluster"
        },
        {
            "data": {"id": "semantic", "label": "Semantic Topics"},
            "classes": "cluster"
        }
    ]

    # assign each node to a container
    for n, d in G.nodes(data=True):
        parent = "fanout" if d["type"] in ("seed", "taxonomy") else "semantic"
        elements.append({
            "data": {"id": n, "label": d["label"], "parent": parent}
        })

    # add edges
    for u, v in G.edges():
        elements.append({"data": {"source": u, "target": v}})

    # define styles
    stylesheet = [
        {
            "selector": ".cluster",
            "style": {
                "shape": "roundrectangle",
                "background-opacity": 0.1,
                "label": "data(label)",
                "text-valign": "top",
                "text-halign": "center",
                "padding": "10px",
                "font-size": "16px"
            }
        },
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-wrap": "wrap",
                "width": "label",
                "height": "label",
                "padding": "5px",
                "font-size": "12px"
            }
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier"
            }
        }
    ]

    # render with CytoscapeJS
    st_cytoscape(
        elements=elements,
        stylesheet=stylesheet,
        layout={
            "name": "cose",
            "idealEdgeLength": 100,
            "nodeRepulsion": 300000
        },
        style={"width": "100%", "height": "650px"},
        key="kg_graph"
    )
