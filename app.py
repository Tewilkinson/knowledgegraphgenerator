# app.py

import streamlit as st
import requests, re
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── CONFIG ──────────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
WD_PREDICATES   = ["P279","P31","P361","P527","P921"]

# ─── UI ───────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Hybrid GPT→Wikidata Knowledge Graph")

with st.sidebar:
    seed       = st.text_input("Seed term", "data warehouse")
    gpt_count  = st.slider("GPT queries on seed", 5, 100, 50)
    max_depth  = st.slider("Wikidata depth under GPT nodes", 1, 5, 3)
    build      = st.button("Build Graph")

# legend
st.markdown(
    "<span style='color:#1f78b4;'>🔵</span>Seed  "
    "<span style='color:#fc8d62;'>🟣</span>GPT  "
    "<span style='color:#66c2a5;'>🟢</span>Wikidata",
    unsafe_allow_html=True
)

# ─── HELPERS ─────────────────────────────────────────────
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
    out = []
    for ln in resp.choices[0].message.content.splitlines():
        ln = ln.strip()
        if not ln: continue
        out.append(re.sub(r"^[-•\s]+", "", ln))
    return out

@st.cache_data
def search_qid(label: str) -> str | None:
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
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    pred_vals = " ".join(f"wdt:{p}" for p in WD_PREDICATES)
    sparql.setQuery(f"""
      SELECT ?pLabel ?objLabel WHERE {{
        VALUES ?p {{ {pred_vals} }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    rows = sparql.query().convert()["results"]["bindings"]
    return [(r["pLabel"]["value"], r["objLabel"]["value"]) for r in rows]

# ─── BUILD GRAPH ──────────────────────────────────────────
def build_graph(seed, gpt_count, max_depth):
    G = nx.DiGraph()
    # A) seed node
    G.add_node(seed, label=seed, source="seed", depth=0)

    # B) seed’s immediate Wikidata neighbors (no recursion)
    seed_qid = search_qid(seed)
    if seed_qid:
        for pred, obj_lbl in fetch_wikidata(seed_qid):
            G.add_node(obj_lbl, label=obj_lbl, source="wikidata", depth=1)
            G.add_edge(seed, obj_lbl, predicate=pred)

    # C) GPT cloud around the seed
    gpt_nodes = get_seed_related_queries(seed, gpt_count)
    for q in gpt_nodes:
        G.add_node(q, label=q, source="gpt", depth=1)
        G.add_edge(seed, q, predicate="related_query")

    # D) recursion ONLY on GPT nodes (not on the seed’s WD neighbors)
    def recurse_wikidata(label, depth):
        if depth >= max_depth:
            return
        qid = search_qid(label)
        if not qid:
            return
        for pred, obj_lbl in fetch_wikidata(qid):
            if not G.has_node(obj_lbl):
                G.add_node(obj_lbl, label=obj_lbl, source="wikidata", depth=depth+2)
            G.add_edge(label, obj_lbl, predicate=pred)
            recurse_wikidata(obj_lbl, depth+1)

    for q in gpt_nodes:
        recurse_wikidata(q, depth=1)

    return G

# ─── RENDER ───────────────────────────────────────────────
def draw_pyvis(G):
    net = Network(height="700px", width="100%", notebook=False)
    for nid, data in G.nodes(data=True):
        color = {
            "seed":"#1f78b4",
            "gpt":"#fc8d62",
            "wikidata":"#66c2a5"
        }[data["source"]]
        net.add_node(nid,
                     label=data["label"],
                     title=f"{data['source']} depth={data['depth']}",
                     color=color)
    for u,v,d in G.edges(data=True):
        net.add_edge(u, v, title=d.get("predicate",""))
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── MAIN ─────────────────────────────────────────────────
if build:
    with st.spinner("Building hybrid graph…"):
        G = build_graph(seed, gpt_count, max_depth)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    st.components.v1.html(draw_pyvis(G), height=750, scrolling=True)
