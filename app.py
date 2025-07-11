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

# --- 2. QID RESOLUTION ---
@st.cache_data(show_spinner=False)
def search_qid(label: str) -> str | None:
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
    return results[0]["id"] if results else None

# --- 3. FETCH RELATIONS ---
@st.cache_data(show_spinner=False)
def fetch_wikidata_relations(qid: str, predicates: list[str]) -> list[tuple[str,str,str]]:
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

# --- 4. EMBEDDINGS & SIMILARITY ---
@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(resp.data[0].embedding)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- 5. BUILD / EXPAND GRAPH ---
def expand_node(
    G: nx.DiGraph, 
    qid: str, 
    label: str, 
    depth: int = 0, 
    max_depth: int = 2
):
    # 1) Prevent re-visiting
    if G.has_node(qid):
        return

    # 2) Add this node with its metadata
    emb = get_embedding(label)
    G.add_node(qid, label=label, embedding=emb.tolist(), depth=depth)

    # 3) If we've hit our depth cap, stop before fetching edges
    if depth >= max_depth:
        return

    # 4) Otherwise, fetch relations and recurse
    relations = fetch_wikidata_relations(qid, predicates=["P279","P31","P361"])
    for pred_label, obj_qid, obj_label in relations:
        G.add_edge(qid, obj_qid, predicate=pred_label)
        expand_node(G, obj_qid, obj_label, depth + 1, max_depth)

# --- 6. VISUALIZATION ---
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="600px", width="100%", notebook=False)
    # nodes
    for node_id, data in G.nodes(data=True):
        lbl = data.get("label", node_id)
        title = f"Depth: {data.get('depth', 'N/A')}"
        net.add_node(node_id, label=lbl, title=title)
    # edges
    for src, dst, data in G.edges(data=True):
        net.add_edge(src, dst, title=data.get("predicate", ""))
    # allow physics controls
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# --- 7. STREAMLIT APP ---
st.set_page_config(layout="wide")
st.title("🔎 Interactive Wikidata Knowledge Graph")

seed      = st.text_input("Enter an entity name", value="data warehouse")
max_depth = st.slider("Max crawl depth", 1, 4, 2)

if st.button("🔍 Build Graph"):
    with st.spinner("Resolving QID…"):
        qid = search_qid(seed)
    if not qid:
        st.error(f"No Wikidata item found for “{seed}”. Try a different term.")
    else:
        st.success(f"Found QID: {qid}")
        with st.spinner("Building graph…"):
            G = nx.DiGraph()
            expand_node(G, qid, seed, depth=0, max_depth=max_depth)

        st.info(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        html = draw_pyvis(G)
        st.components.v1.html(html, height=700, scrolling=True)
