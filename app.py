# app.py

import streamlit as st
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
import numpy as np
import networkx as nx
from pyvis.network import Network
import requests
import re

# --- 1. CONFIG ---
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_API = "https://www.wikidata.org/w/api.php"

# --- 2. QID LOOKUP ---
@st.cache_data(show_spinner=False)
def search_qid(label: str) -> str | None:
    resp = requests.get(
        WIKIDATA_SEARCH_API,
        params={
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": 1
        }
    ).json()
    hits = resp.get("search", [])
    return hits[0]["id"] if hits else None

# --- 3. WIKIDATA RELATIONS ---
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
    data = sparql.query().convert()["results"]["bindings"]
    return [
        (
            b["pLabel"]["value"],
            b["obj"]["value"].rsplit("/", 1)[-1],
            b["objLabel"]["value"]
        )
        for b in data
    ]

# --- 4. GPT “RELATED SUBTOPICS” ---
@st.cache_data(show_spinner=False)
def get_related_topics(label: str, n: int = 5) -> list[str]:
    prompt = (
        f"List {n} DISTINCT, concise subtopics of “{label}” "
        "as a bullet list, each on its own line. "
        "Return ONLY the subtopic names."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        clean = re.sub(r"^[-•\d\.\)\s]+", "", ln)
        if clean:
            lines.append(clean)
    return lines

# --- 5. OPENAI EMBEDDINGS ---
@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return np.array(resp.data[0].embedding)

# --- 6. GRAPH EXPANSION ---
def expand_node(
    G: nx.DiGraph,
    qid: str,
    label: str,
    depth: int,
    max_depth: int,
    subtopics_per_node: int
):
    if G.has_node(qid):
        return

    # Add node metadata
    G.add_node(qid, label=label, depth=depth)

    # Stop if at max depth
    if depth >= max_depth:
        return

    # 1) Wikidata relations
    for pred, obj_qid, obj_label in fetch_wikidata_relations(
        qid, predicates=["P279", "P31", "P361"]
    ):
        G.add_edge(qid, obj_qid, predicate=pred)
        expand_node(G, obj_qid, obj_label, depth + 1, max_depth, subtopics_per_node)

    # 2) GPT “related_to” subtopics
    for sub in get_related_topics(label, n=subtopics_per_node):
        sub_qid = search_qid(sub)
        if not sub_qid:
            continue
        G.add_edge(qid, sub_qid, predicate="related_to")
        expand_node(G, sub_qid, sub, depth + 1, max_depth, subtopics_per_node)

# --- 7. VISUALIZATION WITH PYVIS ---
def draw_pyvis(G: nx.DiGraph) -> str:
    net = Network(height="650px", width="100%", notebook=False)

    # Safely compute max depth
    depths = [data.get("depth", 0) for _, data in G.nodes(data=True)]
    maxd = max(depths) if depths else 0

    # Add nodes
    for node_id, data in G.nodes(data=True):
        d = data.get("depth", 0)
        lbl = data.get("label", node_id)
        # color by depth
        r = int(255 * d / maxd) if maxd else 0
        g = int(200 * (maxd - d) / maxd) if maxd else 200
        color = f"rgba({r}, {g}, 150, 0.8)"
        net.add_node(
            node_id,
            label=lbl,
            title=f"Depth: {d}",
            color=color
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data.get("predicate", ""))

    net.show_buttons(filter_=["physics"])
    return net.generate_html()

# --- 8. STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🔎 Deep Interactive Wikidata Knowledge Graph")

with st.sidebar:
    seed = st.text_input("Seed entity", value="data warehouse")
    max_depth = st.slider("Max crawl depth", 1, 5, 3)
    subtopics_per_node = st.slider("GPT subtopics/node", 1, 8, 5)
    build = st.button("🔍 Build Graph")

if build:
    with st.spinner("Resolving QID…"):
        root_qid = search_qid(seed)
    if not root_qid:
        st.error(f"No QID found for “{seed}”. Try another term.")
    else:
        st.success(f"Found QID: {root_qid}")
        G = nx.DiGraph()
        with st.spinner("Expanding graph…"):
            expand_node(
                G,
                root_qid,
                seed,
                depth=0,
                max_depth=max_depth,
                subtopics_per_node=subtopics_per_node
            )
        st.info(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")
        html = draw_pyvis(G)
        st.components.v1.html(html, height=750, scrolling=True)
