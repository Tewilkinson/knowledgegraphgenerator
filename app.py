# app.py

import streamlit as st
import requests, re
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── 1. CONFIG ────────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"

# Which predicates to fetch on every Wikidata node:
WD_PREDICATES = ["P279",  # subclass of
                 "P31",   # instance of
                 "P361",  # part of
                 "P527",  # has part
                 "P921"   # main subject
                ]

# ─── 2. UI ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Hybrid GPT→Wikidata Knowledge Graph")

with st.sidebar:
    seed       = st.text_input("Seed term", "data warehouse")
    gpt_count  = st.slider("GPT queries on seed", 5, 100, 50)
    max_depth  = st.slider("Wikidata depth",    1,   5,   3)
    build      = st.button("Build Graph")

# Legend
st.markdown(
    "<span style='color:#1f78b4;'>🔵</span>Seed  "
    "<span style='color:#fc8d62;'>🟣</span>GPT  "
    "<span style='color:#66c2a5;'>🟢</span>Wikidata",
    unsafe_allow_html=True
)

# ─── 3. HELPERS ────────────────────────────────────────────
def get_seed_related_queries(term: str, n: int) -> list[str]:
    prompt = (
        f"Give me {n} concise, distinct search queries related to “{term}”. "
        "Return them as a bulleted list, one query per line."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role":"system","content":"You are a helpful assistant."},
          {"role":"user","content":prompt}
        ],
        temperature=0.7
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
    """Lookup via Wikidata API, fallback to None."""
    try:
        r = requests.get(WIKIDATA_API, params={
            "action":"wbsearchentities",
            "search":label,
            "language":"en",
            "format":"json",
            "limit":1
        }, timeout=5).json()
        return r.get("search", [{}])[0].get("id")
    except:
        return None

@st.cache_data
def fetch_wikidata(qid: str) -> list[tuple[str,str]]:
    """
    Returns a list of (predicateLabel, objectLabel) for WD_PREDICATES.
    """
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    preds   = " ".join(f"wdt:{p}" for p in WD_PREDICATES)
    sparql.setQuery(f"""
      SELECT ?pLabel ?objLabel WHERE {{
        VALUES ?p {{ {preds} }}
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
def build_graph(seed: str, gpt_count: int, max_depth: int):
    G = nx.DiGraph()

    # A) Add seed node
    G.add_node(seed, label=seed, source="seed", depth=0)

    # B) Pull immediate Wikidata relations off the seed
    seed_qid = search_qid(seed)
    wiki_frontier = []
    if seed_qid:
        for pred, obj_lbl in fetch_wikidata(seed_qid):
            G.add_node(obj_lbl, label=obj_lbl, source="wikidata", depth=1)
            G.add_edge(seed, obj_lbl, predicate=pred)
            wiki_frontier.append((obj_lbl, 1))

    # C) Pull the big GPT cloud around the seed
    gpt_neighbors = get_seed_related_queries(seed, gpt_count)
    for q in gpt_neighbors:
        G.add_node(q, label=q, source="gpt", depth=1)
        G.add_edge(seed, q, predicate="related_query")

    # D) Recursively expand *all* Wikidata nodes (seed's + GPT's) to `max_depth`
    def recurse(label: str, depth: int):
        if depth >= max_depth:
            return
        qid = search_qid(label)
        if not qid:
            return
        for pred, obj_lbl in fetch_wikidata(qid):
            if not G.has_node(obj_lbl):
                G.add_node(obj_lbl, label=obj_lbl, source="wikidata", depth=depth+1)
            G.add_edge(label, obj_lbl, predicate=pred)
            recurse(obj_lbl, depth+1)

    # Seed frontier + GPT neighbors
    for lbl, d in wiki_frontier:
        recurse(lbl, d)
    for q in gpt_neighbors:
        recurse(q, 1)

    return G

# ─── 5. VISUALIZE ─────────────────────────────────────────
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="700px", width="100%", notebook=False)
    for nid, data in G.nodes(data=True):
        src   = data["source"]
        color = {"seed":"#1f78b4","gpt":"#fc8d62","wikidata":"#66c2a5"}[src]
        net.add_node(
            nid,
            label=data["label"],
            title=f"{src} (depth {data['depth']})",
            color=color
        )
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, title=d.get("predicate",""))
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 6. MAIN ───────────────────────────────────────────────
if build:
    with st.spinner("Building hybrid graph…"):
        G = build_graph(seed, gpt_count, max_depth)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    st.components.v1.html(draw_pyvis(G), height=750, scrolling=True)
