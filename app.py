# app.py

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
import numpy as np
import networkx as nx
from pyvis.network import Network
import requests
import re

# ─── 1. CONFIG ─────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL    = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_API= "https://www.wikidata.org/w/api.php"

# Fixed depth & GPT subtopics per node
MAX_DEPTH = 4
GPT_TOPICS = 5

# ─── 2. QID LOOKUP ─────────────────────────────────────
@st.cache_data
def search_qid(label: str) -> str | None:
    r = requests.get(
        WIKIDATA_SEARCH_API,
        params={
            "action":"wbsearchentities",
            "search": label,
            "language":"en",
            "format":"json",
            "limit":1
        }
    ).json()
    hits = r.get("search", [])
    return hits[0]["id"] if hits else None

# ─── 3. FETCH WIKIDATA RELATIONS ────────────────────────
@st.cache_data
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
    rows = sparql.query().convert()["results"]["bindings"]
    return [
        (
          b["pLabel"]["value"],
          b["obj"]["value"].rsplit("/",1)[-1],
          b["objLabel"]["value"]
        )
        for b in rows
    ]

# ─── 4. GPT “RELATED SUBTOPICS” ─────────────────────────
@st.cache_data
def get_related_topics(label: str, n: int=GPT_TOPICS) -> list[str]:
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
        if not ln: continue
        clean = re.sub(r"^[-•\d\.\)\s]+","",ln)
        if clean: out.append(clean)
    return out

# ─── 5. EMBEDDINGS (for future clustering if desired) ────
@st.cache_data
def get_embedding(text: str) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(resp.data[0].embedding)

# ─── 6. RECURSIVE EXPANSION ─────────────────────────────
def expand_node(
    G: nx.DiGraph,
    qid: str,
    label: str,
    depth: int
):
    # avoid repeats
    if G.has_node(qid):
        return

    # store label & depth
    G.add_node(qid, label=label, depth=depth)

    # stop if max depth
    if depth >= MAX_DEPTH:
        return

    # 1) Wikidata edges
    for pred, obj_qid, obj_label in fetch_wikidata_relations(
        qid, predicates=["P279","P31","P361"]
    ):
        G.add_edge(qid, obj_qid, predicate=pred)
        expand_node(G, obj_qid, obj_label, depth+1)

    # 2) GPT subtopics
    for sub in get_related_topics(label):
        sub_qid = search_qid(sub)
        # If we got a QID, link; if not, still create a node with a synthetic ID
        node_id = sub_qid or f"GPT:{sub}"
        G.add_node(node_id, label=sub, depth=depth+1)
        G.add_edge(qid, node_id, predicate="related_to")
        expand_node(G, node_id, sub, depth+1)

# ─── 7. RENDER WITH PYVIS ────────────────────────────────
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="700px", width="100%", notebook=False)
    # safe max depth
    depths = [d.get("depth",0) for _,d in G.nodes(data=True)]
    maxd = max(depths) if depths else 0

    for nid, data in G.nodes(data=True):
        d = data.get("depth",0)
        lbl = data.get("label",nid)
        # color gradient by depth
        r = int(255 * d/(maxd or 1))
        g = int(200 * (maxd-d)/(maxd or 1))
        net.add_node(nid, label=lbl, title=f"Depth: {d}",
                     color=f"rgba({r},{g},150,0.8)")

    for u,v,data in G.edges(data=True):
        net.add_edge(u,v,title=data.get("predicate",""))

    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 8. STREAMLIT UI ─────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Deep Interactive Wikidata Knowledge Graph")

seed = st.text_input("🔍 Seed entity", value="data warehouse")
if st.button("Build Deep Graph"):
    # 1) find root QID
    qid = search_qid(seed)
    if not qid:
        st.error(f"No match on Wikidata for “{seed}”")
        st.stop()

    st.success(f"Resolved “{seed}” → {qid}")
    G = nx.DiGraph()
    with st.spinner("Expanding… this may take a moment"]:
        expand_node(G, qid, seed, depth=0)

    st.info(f"🚀 Built graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    html = draw_pyvis(G)
    st.components.v1.html(html, height=750, scrolling=True)
