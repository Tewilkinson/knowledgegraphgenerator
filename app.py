# app.py

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
import numpy as np
import networkx as nx
from pyvis.network import Network

# --- 1. CONFIG ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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
    """
    Calls the OpenAI v1 client to get a single embedding vector.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",   # or e.g. "text-embedding-ada-002"
        input=[text]                      # note: list of strings
    )
    vec = resp.data[0].embedding
    return np.array(vec)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- 4. BUILD/EXPAND GRAPH ---
def expand_node(G: nx.DiGraph, qid: str, label: str, depth: int=0, max_depth: int=2):
    """
    Recursively adds nodes & edges from Wikidata + embeddings enrichment.
    """
    if depth > max_depth or G.has_node(qid):
        return
    emb = get_embedding(label)
    G.add_node(qid, label=label, embedding=emb.tolist(), depth=depth)

    # fetch Wikidata relations
    relations = fetch_wikidata_relations(qid, predicates=["P279","P31","P361"])
    for pred_label, obj_qid, obj_label in relations:
        G.add_edge(qid, obj_qid, predicate=pred_label)
        expand_node(G, obj_qid, obj_label, depth+1, max_depth)

# --- 5. VISUALIZATION ---
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="600px", width="100%", notebook=False)
    for node_id, data in G.nodes(data=True):
        net.add_node(node_id, label=data["label"], title=f"Depth: {data['depth']}")
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, title=data["predicate"])
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# --- 6. STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🔎 Interactive Wikidata Knowledge Graph")

# 6.1 Seed input & build trigger
seed = st.text_input("Enter an entity name (e.g. ‘Data warehouse’)", value="Data warehouse")
max_depth = st.slider("Max crawl depth", 1, 4, 2)
if st.button("🔍 Build Graph"):
    # ----- Resolve seed → QID (simple SPARQL search) -----
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.setQuery(f"""
    SELECT ?item WHERE {{
      ?item rdfs:label "{seed}"@en .
    }} LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    res = sparql.query().convert()["results"]["bindings"]
    if not res:
        st.error(f"No Wikidata item found for “{seed}”")
    else:
        qid = res[0]["item"]["value"].split("/")[-1]

        # ----- Build & expand graph -----
        G = nx.DiGraph()
        expand_node(G, qid, seed, depth=0, max_depth=max_depth)

        # ----- Render -----
        html = draw_pyvis(G)
        st.components.v1.html(html, height=700, scrolling=True)
