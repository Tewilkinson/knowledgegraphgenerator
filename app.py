import streamlit as st
import requests
import networkx as nx
from st_cytoscape import cytoscape
# no more streamlit_cytoscape import!

# 1) LOOKUP: label → Wikidata QID
@st.cache_data
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
    return resp.json()["search"][0]["id"]

# 2) TAXONOMY: subclasses (fan-out)
@st.cache_data
def get_subclasses(qid: str, limit: int = 20):
    sparql = f"""
    SELECT ?child ?childLabel WHERE {{
      ?child wdt:P279 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    rows = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": sparql},
        headers={"Accept": "application/sparql-results+json"}
    ).json()["results"]["bindings"]
    return [
        (r["child"]["value"].rsplit("/",1)[-1], r["childLabel"]["value"])
        for r in rows
    ]

# 3) SEMANTIC: ConceptNet neighbours
@st.cache_data
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    data = requests.get(
        f"http://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}
    ).json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0))
        for e in data
    ]

# 4) BUILD the hybrid graph
def build_graph(seed: str, depth: int, tax_lim: int, sem_lim: int):
    qid = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(qid, label=seed, type="seed")

    # fan-out
    for cq, cl in get_subclasses(qid, tax_lim):
        G.add_node(cq, label=cl, type="taxonomy")
        G.add_edge(qid, cq)

    # semantic
    for lbl, _ in get_conceptnet_neighbors(seed, sem_lim):
        node_id = f"CN:{lbl}"
        G.add_node(node_id, label=lbl, type="semantic")
        G.add_edge(qid, node_id)

    # 2-hop fan-out
    if depth > 1:
        for n,d in list(G.nodes(data=True)):
            if d["type"]=="taxonomy":
                for cq,cl in get_subclasses(n, tax_lim//2):
                    G.add_node(cq, label=cl, type="taxonomy")
                    G.add_edge(n, cq)

    return G

# 5) STREAMLIT UI with st-cytoscape
st.set_page_config(layout="wide")
st.title("🔗 Grouped Knowledge-Graph Clusters")
seed   = st.text_input("Seed topic", "data warehouse")
depth  = st.slider("Taxonomy depth", 1, 2, 1)
tax_lim= st.slider("Max subclasses",  5, 50, 20)
sem_lim= st.slider("Max neighbours",  5, 50, 20)

if st.button("Generate Graph"):
    G = build_graph(seed, depth, tax_lim, sem_lim)

    # prepare Cytoscape elements and clusters
    elements = [
        {"data":{"id":"taxonomy_group","label":"Fan-out Topics"},"classes":"cluster"},
        {"data":{"id":"semantic_group","label":"Semantic Topics"},"classes":"cluster"},
    ]
    for n,d in G.nodes(data=True):
        parent = "taxonomy_group" if d["type"] in ("seed","taxonomy") else "semantic_group"
        elements.append({"data":{"id":n,"label":d["label"],"parent":parent}})
    for u,v in G.edges():
        elements.append({"data":{"source":u,"target":v}})

    stylesheet = [
        {"selector":".cluster","style":{
            "shape":"roundrectangle","background-color":"#EEE",
            "text-valign":"top","text-halign":"center","padding":"10px"
         }},
        {"selector":"node","style":{
            "label":"data(label)","text-wrap":"wrap",
            "width":"label","height":"label","padding":"5px"
         }},
        {"selector":"edge","style":{"curve-style":"bezier"}}
    ]

    cytoscape(
        elements,
        stylesheet,
        layout={
          "name":"cose",
          "idealEdgeLength":100,
          "nodeRepulsion":400000
        },
        style={"width":"100%","height":"650px"},
        key="graph"
    )
