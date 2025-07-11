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
    seed       = st.text_input("Seed entity", "data warehouse")
    max_depth  = st.slider("Max crawl depth", 1, 5, 5)
    gpt_topics = st.slider("GPT topics/node", 1, 30, 10)
    build      = st.button("Build Graph")

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
    Try the wbsearchentities API, then fall back to a SPARQL label match.
    """
    # 3A) API call
    try:
        r = requests.get(
            WIKIDATA_SEARCH,
            params={
                "action":"wbsearchentities",
                "search":label,
                "language":"en",
                "format":"json",
                "limit":1
            },
            timeout=5
        )
        r.raise_for_status()
        hits = r.json().get("search", [])
        if hits:
            return hits[0]["id"]
    except Exception:
        pass

    # 3B) SPARQL fallback on rdfs:label
    try:
        sparql = SPARQLWrapper(WIKIDATA_SPARQL)
        sparql.setQuery(f"""
          PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
          SELECT ?item WHERE {{
            ?item rdfs:label "{label}"@en .
          }} LIMIT 1
        """)
        sparql.setReturnFormat(JSON)
        res = sparql.query().convert()["results"]["bindings"]
        if res:
            return res[0]["item"]["value"].rsplit("/",1)[-1]
    except Exception:
        pass

    return None


@st.cache_data
def fetch_wikidata_relations(qid: str, preds: list[str]) -> list[tuple[str,str,str]]:
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    vals   = " ".join(f"wdt:{p}" for p in preds)
    sparql.setQuery(f"""
      SELECT ?pLabel ?obj ?objLabel WHERE {{
        VALUES ?p {{ {vals} }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    bindings = sparql.query().convert()["results"]["bindings"]
    return [
        (
          b["pLabel"]["value"],
          b["obj"]["value"].rsplit("/",1)[-1],
          b["objLabel"]["value"]
        )
        for b in bindings
    ]


@st.cache_data
def get_related_topics(label: str, n: int) -> list[str]:
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
    items = []
    for line in text.splitlines():
        clean = re.sub(r"^[-•\d\.\)\s]+","", line.strip())
        if clean:
            items.append(clean)
    return items

# ─── 4. GRAPH EXPANSION ────────────────────────────────────
def expand_node(G: nx.DiGraph, qid: str, label: str, depth: int):
    if G.has_node(qid):
        return

    source = "wikidata" if qid.startswith("Q") else "gpt"
    G.add_node(qid, label=label, depth=depth, source=source)

    if depth >= max_depth:
        return

    # Wikidata edges
    if source == "wikidata":
        for prop, obj_q, obj_lbl in fetch_wikidata_relations(
            qid, ["P279","P31","P361"]
        ):
            G.add_edge(qid, obj_q, predicate=prop)
            expand_node(G, obj_q, obj_lbl, depth+1)

    # GPT “related_to” edges
    for sub in get_related_topics(label, gpt_topics):
        sub_q = search_qid(sub)
        node_id = sub_q or f"GPT:{sub}"
        G.add_edge(qid, node_id, predicate="related_to")
        expand_node(G, node_id, sub, depth+1)

# ─── 5. VISUALIZATION ─────────────────────────────────────
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="700px", width="100%", notebook=False)

    for nid, data in G.nodes(data=True):
        lbl    = data.get("label", nid)
        src    = data.get("source", "wikidata")
        color  = "#66c2a5" if src == "wikidata" else "#fc8d62"
        title  = f"{lbl} (depth {data.get('depth', 0)})"
        net.add_node(nid, label=lbl, title=title, color=color)

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, title=d.get("predicate",""))

    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 6. MAIN ───────────────────────────────────────────────
if build:
    with st.spinner("Resolving seed to QID…"):
        root = search_qid(seed)

    if not root:
        st.error(f"❌ No Wikidata match for “{seed}”.")
        st.stop()

    st.success(f"✅ Seed → QID {root}")
    G = nx.DiGraph()
    with st.spinner("Expanding graph… this may take a while"):
        expand_node(G, root, seed, depth=0)

    st.info(f"🚀 Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    html = draw_pyvis(G)
    st.components.v1.html(html, height=750, scrolling=True)
