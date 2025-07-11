# app.py

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
import numpy as np
import networkx as nx
from pyvis.network import Network
import requests, re

# ─── 1. CONFIG & CLIENTS ─────────────────────────────────
openai_client     = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL   = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH   = "https://www.wikidata.org/w/api.php"

# ─── 2. UI CONTROLS ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Deep Interactive Wikidata Knowledge Graph")

with st.sidebar:
    seed            = st.text_input("Seed entity", "data warehouse")
    max_depth       = st.slider("Max crawl depth", 1, 5, 3)
    gpt_topics      = st.slider("GPT topics/node", 1, 30, 8)
    build           = st.button("Build Graph")

# Inline legend
st.markdown(
    "<span style='display:inline-block;width:12px;height:12px;"
    "background-color:#66c2a5;margin-right:6px;'></span>Wikidata&nbsp;&nbsp;"
    "<span style='display:inline-block;width:12px;height:12px;"
    "background-color:#fc8d62;margin-right:6px;'></span>GPT",
    unsafe_allow_html=True
)

# ─── 3. HELPERS ────────────────────────────────────────────
@st.cache_data
def search_qid(label: str) -> str | None:
    """
    Lookup label via Wikidata search API.
    Returns QID or None on error / no hit.
    """
    try:
        resp = requests.get(
            WIKIDATA_SEARCH,
            params={
                "action": "wbsearchentities",
                "search": label,
                "language": "en",
                "format": "json",
                "limit": 1
            },
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("search", [])
        return hits[0]["id"] if hits else None
    except (requests.RequestException, ValueError):
        # network error, bad JSON, etc.
        return None

@st.cache_data
def fetch_wikidata_relations(qid: str, preds: list[str]):
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    pvals   = " ".join(f"wdt:{p}" for p in preds)
    sparql.setQuery(f"""
      SELECT ?pLabel ?obj ?objLabel WHERE {{
        VALUES ?p {{ {pvals} }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    rows = sparql.query().convert()["results"]["bindings"]
    return [
        (r["pLabel"]["value"],
         r["obj"]["value"].rsplit("/",1)[-1],
         r["objLabel"]["value"])
      for r in rows
    ]

@st.cache_data
def get_related_topics(label: str, n: int):
    prompt = (
        f"List {n} DISTINCT, concise subtopics of “{label}” "
        "as a bullet list. Return ONLY the subtopic names."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: 
            continue
        clean = re.sub(r"^[-•\d\.\)\s]+","", ln)
        if clean:
            out.append(clean)
    return out

# ─── 4. GRAPH EXPANSION ────────────────────────────────────
def expand_node(G: nx.DiGraph, qid: str, label: str, depth: int):
    # If node exists, ensure label/depth are set
    if G.has_node(qid):
        meta = G.nodes[qid]
        meta.setdefault("label", label)
        meta.setdefault("depth", depth)
        return

    source = "wikidata" if qid.startswith("Q") else "gpt"
    G.add_node(qid, label=label, depth=depth, source=source)

    if depth >= max_depth:
        return

    # Wikidata relations
    if source == "wikidata":
        for pred, obj_id, obj_lbl in fetch_wikidata_relations(
            qid, ["P279","P31","P361"]
        ):
            G.add_edge(qid, obj_id, predicate=pred)
            expand_node(G, obj_id, obj_lbl, depth+1)

    # GPT subtopics
    for sub in get_related_topics(label, gpt_topics):
        sub_qid = search_qid(sub)
        node_id = sub_qid or f"GPT:{sub}"
        G.add_edge(qid, node_id, predicate="related_to")
        expand_node(G, node_id, sub, depth+1)

# ─── 5. VISUALIZATION ─────────────────────────────────────
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="700px", width="100%", notebook=False)
    for nid, data in G.nodes(data=True):
        lbl   = data.get("label", nid)
        color = "#66c2a5" if data.get("source")=="wikidata" else "#fc8d62"
        title = f"{lbl} (depth {data.get('depth')})"
        net.add_node(nid, label=lbl, title=title, color=color)
    for u,v,data in G.edges(data=True):
        net.add_edge(u, v, title=data.get("predicate",""))
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 6. MAIN ───────────────────────────────────────────────
if build:
    with st.spinner("Resolving seed to QID…"):
        root_qid = search_qid(seed)
    if not root_qid:
        st.error(f"No Wikidata match for “{seed}”.")
        st.stop()

    st.success(f"Seed → QID {root_qid}")
    G = nx.DiGraph()
    with st.spinner("Expanding graph… this may take a while"):
        expand_node(G, root_qid, seed, depth=0)

    st.info(f"🚀 Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    html = draw_pyvis(G)
    st.components.v1.html(html, height=750, scrolling=True)
