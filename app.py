# requirements: streamlit, SPARQLWrapper, openai, networkx, pyvis, numpy

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
import openai, numpy as np, networkx as nx
from pyvis.network import Network

# --- 1. CONFIG ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# --- 2. FETCH RELATIONS FROM WIKIDATA ---
@st.cache_data(show_spinner=False)
def fetch_wikidata_relations(qid: str, predicates: list[str]) -> list[tuple[str,str,str]]:
    """
    Returns a list of (predicate_label, obj_qid, obj_label).
    """
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    preds = " ".join(f"wdt:{p}" for p in predicates)
    sparql.setQuery(f"""
    SELECT ?pLabel ?obj ?objLabel WHERE {{
      VALUES ?p {{ {preds} }}
      wd:{qid} ?p ?obj .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return [
      (r["pLabel"]["value"], 
       r["obj"]["value"].split("/")[-1], 
       r["objLabel"]["value"])
      for r in results
    ]

# --- 3. GET EMBEDDINGS & SIMILARITY ---
@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return np.array(resp["data"][0]["embedding"])

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- 4. BUILD/EXPAND GRAPH ---
def expand_node(G: nx.DiGraph, qid: str, label: str, depth: int=0, max_depth: int=2):
    if depth >= max_depth or G.has_node(qid):
        return
    embedding = get_embedding(label)
    G.add_node(qid, label=label, embedding=embedding.tolist(), depth=depth)
    
    # fetch Wikidata relations
    relations = fetch_wikidata_relations(qid, predicates=["P279","P31","P361"])
    for pred_label, obj_qid, obj_label in relations:
        G.add_edge(qid, obj_qid, predicate=pred_label)
        expand_node(G, obj_qid, obj_label, depth+1, max_depth)

# --- 5. VISUALIZATION ---
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="600px", width="100%", notebook=False)
    for n, data in G.nodes(data=True):
        net.add_node(n, label=data["label"], title=f"Depth: {data['depth']}")
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data["predicate"])
    # force a physics layout
    net.show_buttons(filter_=['physics'])
    html = net.generate_html()
    return html

# --- 6. STREAMLIT APP ---
st.title("🔎 Interactive Wikidata Knowledge Graph")

# 6.1 Seed input & resolve to QID
seed = st.text_input("Enter an entity name (e.g. ‘Data warehouse’)", value="Data warehouse")
if st.button("🔍 Build Graph"):
    # 1) resolve seed to QID via Wikidata API
    # (You can hit the Wikidata Search API or SPARQL to find QID for the label)
    # For brevity, let’s assume we found qid:
    qid = "Q11707"  # example: Data warehouse
    
    # 2) build graph
    G = nx.DiGraph()
    expand_node(G, qid, seed, depth=0, max_depth=2)
    
    # 3) visualize
    html = draw_pyvis(G)
    st.components.v1.html(html, height=650, scrolling=True)
