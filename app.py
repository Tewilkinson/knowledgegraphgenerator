# app.py

import streamlit as st
import requests, re
import networkx as nx
import numpy as np
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── 1. CONFIG ────────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"

# ─── 2. UI ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Deep Hybrid GPT + Wikidata Graph")

with st.sidebar:
    seed       = st.text_input("Seed term", "data warehouse")
    max_depth  = st.slider("Wikidata depth", 1, 5, 3)
    build      = st.button("Build Graph")

# Legend
st.markdown(
    """
    <span style="display:inline-block;
                 width:12px;height:12px;
                 background:#fc8d62;
                 margin-right:6px;"></span>GPT
    &nbsp;&nbsp;
    <span style="display:inline-block;
                 width:12px;height:12px;
                 background:#66c2a5;
                 margin-right:6px;"></span>Wikidata
    """, unsafe_allow_html=True
)

# ─── 3. HELPERS ────────────────────────────────────────────
def get_seed_related_queries(term: str, n: int=50) -> list[str]:
    """One big GPT call for n related queries of the seed."""
    prompt = (
        f"Give me {n} concise, distinct search queries related to “{term}”. "
        "Return them as a bullet list (no numbering)."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7,
    )
    lines = resp.choices[0].message.content.splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        clean = re.sub(r"^[-•\s]+","", ln)
        out.append(clean)
    return out

@st.cache_data
def search_qid(label: str) -> str | None:
    """Simple Wikidata search→QID (we’ll only use it to recurse)."""
    try:
        r = requests.get(WIKIDATA_SEARCH, params={
            "action":"wbsearchentities",
            "search": label,
            "language":"en",
            "format":"json",
            "limit": 1
        }, timeout=5).json()
        return r.get("search", [{}])[0].get("id")
    except:
        return None

@st.cache_data
def fetch_wd(qid: str) -> list[tuple[str,str]]:
    """Get (predicateLabel, objectLabel) for the 3 predicates."""
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.setQuery(f"""
      SELECT ?pLabel ?objLabel WHERE {{
        VALUES ?p {{ wdt:P279 wdt:P31 wdt:P361 }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    rows = sparql.query().convert()["results"]["bindings"]
    return [
      (r["pLabel"]["value"], r["objLabel"]["value"])
      for r in rows
    ]

# ─── 4. BUILD GRAPH ────────────────────────────────────────
def build_graph(seed: str, max_depth: int):
    G = nx.DiGraph()

    # 4A) Seed node
    G.add_node(seed, label=seed, source="seed", depth=0)

    # 4B) GPT level: one big call
    related = get_seed_related_queries(seed, n=50)
    for q in related:
        G.add_node(q, label=q, source="gpt", depth=1)
        G.add_edge(seed, q, predicate="related_query")

    # 4C) Wikidata recursion under each GPT node
    def recurse(label, depth):
        if depth > max_depth:
            return
        qid = search_qid(label)
        if not qid:
            return
        for pred, obj_lbl in fetch_wd(qid):
            # node
            G.add_node(obj_lbl, label=obj_lbl, source="wikidata", depth=depth+2)
            # edge
            G.add_edge(label, obj_lbl, predicate=pred)
            recurse(obj_lbl, depth+1)

    for q in related:
        recurse(q, depth=1)

    return G

# ─── 5. RENDER ─────────────────────────────────────────────
def draw(G):
    net = Network(height="700px", width="100%", notebook=False)
    for nid, d in G.nodes(data=True):
        color = {
            "seed":"#1f78b4",
            "gpt":"#fc8d62",
            "wikidata":"#66c2a5"
        }[d["source"]]
        net.add_node(nid, label=d["label"], title=f"{d['source']} depth={d['depth']}", color=color)
    for u,v,d in G.edges(data=True):
        net.add_edge(u,v,title=d["predicate"])
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 6. MAIN ───────────────────────────────────────────────
if build:
    with st.spinner("Building hybrid graph…"):
        G = build_graph(seed, max_depth)

    st.success(f"Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    html = draw(G)
    st.components.v1.html(html, height=750, scrolling=True)
