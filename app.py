# app.py

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
import numpy as np
import networkx as nx
from pyvis.network import Network
import requests

# --- 1. CONFIG ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# --- 2. QID RESOLUTION VIA WIKIDATA SEARCH API ---
@st.cache_data(show_spinner=False)
def search_qid(label: str) -> str | None:
    """
    Uses the Wikidata API to find the best‐matching QID for a given English label.
    Returns e.g. "Q11707" or None if not found.
    """
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": 1
        }
    ).json()
    results = resp.get("search", [])
    if results:
        return results[0]["id"]
    return None

# --- 3. FETCH RELATIONS FROM WIKIDATA ---
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
    bindings = sparql.query().convert()["results"]["bindings"]
    return [
        (
            b["pLabel"]["value"],
            b["obj"]["value"].rsplit("/", 1)[-1],
            b["objLabel"]["value"]
        )
        for b in bindings
    ]

# --- 4. GET EMBEDDINGS & SIMILARITY ---
@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",  # or "text-embedding-ada-002"
        input=[text]
    )
    vec = resp.data[0].embedding
    return np.array(vec)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- 5. BUILD/EXPAND GRAPH ---
def expand_node(G: nx.DiGraph, qid: str, label: str, depth: int=0, max_depth: int=2):
    if depth > max_depth or G.has_node(qid):
        return
    emb = get_embedding(label)
    G.add_node(qid, label=label, embedding=emb.tolist(), depth=depth)

    relations = fetch_wikidata_relations(qid, predicates=["P279","P31","P361"])
    for pred_label, obj_qid, obj_label in relations:
        G.add_edge(qid, obj_qid, predicate=pred_label)
        expand_node(G, obj_qid, obj_label, depth+1, max_depth)

# --- 6. VISUALIZATION WITH PYVIS ---
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="600px", width="100%", notebook=False)
    for node_id, data in G.nodes(data=True):
        net.add_node(node_id, label=data["label"], title=f"Depth: {data['depth']}")
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, title=data["predicate"])
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# --- 7. STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🔎 Interactive Wikidata Knowledge Graph")

# 7.1 Inputs
seed = st.text_input("Enter an entity name (e.g. ‘Data warehouse’)", value="data warehouse")
max_depth = st.slider("Max crawl depth", 1, 4, 2)

if st.button("🔍 Build Graph"):
    with st.spinner("Resolving QID…"):
        qid = search_qid(seed)
    if not qid:
        st.error(f"No Wikidata item found for “{seed}” – please try a different term.")
    else:
        st.success(f"Found QID: {qid}")
        with st.spinner("Building graph…"):
            G = nx.DiGraph()
            expand_node(G, qid, seed, depth=0, max_depth=max_depth)

        st.info(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        html = draw_pyvis(G)
        st.components.v1.html(html, height=700, scrolling=True)
