import os
import json
import tempfile
from collections import deque

import streamlit as st
import networkx as nx
import community as community_louvain  # pip install python-louvain
from pyvis.network import Network  # pip install pyvis
import openai  # pip install openai

# Page configuration and OpenAI client setup
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_triples(concept: str, max_triples: int = 6):
    # Build prompt without using triple-quoted f-strings
    prompt = (
        f"You’re a KG extractor. Given the concept \"{concept}\", "
        f"list up to {max_triples} related concepts as a JSON array of "
        "[subject, relation, object] triples. "
        "Use relations like \"subclass_of\" or \"related_to\"."
    )
    # Use OpenAI's ChatCompletion API
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content)


def build_graph(seed: str, depth: int = 2, max_triples: int = 6):
    G = nx.DiGraph()
    seen = {seed}
    queue = deque([(seed, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        try:
            triples = extract_triples(node, max_triples)
        except Exception as e:
            st.error(f"Failed to extract triples for '{node}': {e}")
            continue
        for s, rel, o in triples:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, relation=rel)
            if o not in seen:
                seen.add(o)
                queue.append((o, d + 1))
    return G


def detect_communities(G: nx.Graph):
    # Use Louvain on the undirected graph for community detection
    partition = community_louvain.best_partition(G.to_undirected())
    return partition


def visualize_pyvis(G: nx.DiGraph, partition: dict):
    net = Network(height="700px", width="100%", directed=True)
    for node in G.nodes():
        cid = partition.get(node, 0)
        hue = (cid * 60) % 360
        net.add_node(node, label=node, color=f"hsl({hue},70%,50%)")
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        dashed = rel != "subclass_of"
        net.add_edge(u, v, title=rel, dashes=dashed, arrowStrikethrough=False)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.show(tmpfile)
    with open(tmpfile, "r", encoding="utf-8") as f:
        html = f.read()
    return html


def main():
    st.title("🔍 Knowledge Graph Explorer")

    # User inputs
    seed = st.text_input("Enter seed topic", value="Spatial data warehouse")
    depth = st.slider("Expansion depth", 1, 3, 2)
    max_triples = st.slider("Triples per node", 2, 8, 4)

    if st.button("Build Graph"):
        with st.spinner("⏳ Building knowledge graph…"):
            G = build_graph(seed, depth, max_triples)
            if G.number_of_nodes() == 0:
                st.warning("No nodes found. Try a different seed or parameters.")
            else:
                partition = detect_communities(G)
                html = visualize_pyvis(G, partition)
                st.components.v1.html(html, height=700, scrolling=True)


if __name__ == "__main__":
    main()
