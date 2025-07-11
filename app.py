# app.py
import os, json, tempfile
from collections import deque

import streamlit as st
import networkx as nx
import community as community_louvain  # pip install python-louvain
from pyvis.network import Network  # pip install pyvis
from openai import OpenAI

# ── 1) CONFIG ────────────────────────────────────────────────────────────────
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")

# ── 2) TRIPLE EXTRACTION ──────────────────────────────────────────────────────
def extract_triples(concept: str, max_triples: int = 6):
    prompt = f"""
You’re a KG extractor. Given the concept "{concept}",  
list up to {max_triples} related concepts as JSON array of  
[subject, relation, object] triples.  
Use relations like "subclass_of" or "related_to".
""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content)

# ── 3) GRAPH BUILDING ─────────────────────────────────────────────────────────
def build_graph(seed: str, depth: int = 2):
    G = nx.DiGraph()
    seen = {seed}
    queue = deque([(seed, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        try:
            triples = extract_triples(node)
        except Exception as e:
            st.error(f"Failed to extract from “{node}”: {e}")
            continue
        for s, rel, o in triples:
            G.add_node(s); G.add_node(o)
            G.add_edge(s, o, relation=rel)
            if o not in seen:
                seen.add(o)
                queue.append((o, d + 1))
    return G

# ── 4) COMMUNITY DETECTION ────────────────────────────────────────────────────
def detect_communities(G: nx.Graph):
    # use Louvain on the undirected version
    partition = community_louvain.best_partition(G.to_undirected())
    return partition

# ── 5) VISUALIZATION ──────────────────────────────────────────────────────────
def visualize_pyvis(G: nx.DiGraph, partition: dict[int,int]):
    net = Network(height="700px", width="100%", directed=True)
    for node in G.nodes():
        cid = partition.get(node, 0)
        net.add_node(node, label=node, color=f"hsl({cid*60 % 360},70%,50%)")
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data.get("relation", ""), arrowStrikethrough=False)

    # write to a temp HTML and return its contents
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.show(path)
    return open(path, "r", encoding="utf-8").read()

# ── 6) STREAMLIT LAYOUT ───────────────────────────────────────────────────────
st.title("🔍 Knowledge Graph Explorer")

seed = st.text_input("Enter seed topic", value="Spatial data warehouse")
depth = st.slider("Expansion depth", 1, 3, 2)
max_triples = st.slider("Triples per node", 2, 8, 4)

if st.button("Build Graph"):
    with st.spinner("⏳ Building knowledge graph…"):
        G = build_graph(seed, depth)
        if G.number_of_nodes() == 0:
            st.warning("No nodes found. Try a different seed or depth.")
        else:
            partition = detect_communities(G)
            html = visualize_pyvis(G, partition)
            st.components.v1.html(html, height=700, scrolling=True)
