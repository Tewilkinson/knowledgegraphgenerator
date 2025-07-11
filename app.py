import os
import json
import tempfile
from collections import deque

import streamlit as st
# Ensure OpenAI client is installed
try:
    import openai
except ImportError:
    st.error("The 'openai' package is not installed. Please add 'openai' to your requirements.txt or run 'pip install openai'.")
    st.stop()

import networkx as nx
import community as community_louvain  # pip install python-louvain
from pyvis.network import Network  # pip install pyvis

# Page configuration
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_triples(concept: str, max_triples: int = 6):
    """
    Queries the OpenAI API for triples related to a concept and returns a list of [subject, relation, object].
    """
    prompt = (
        f"You are a knowledge graph extractor. Given the concept '{concept}', "
        f"provide up to {max_triples} related concepts as a JSON array of "
        "[subject, relation, object] triples. Use relations 'subclass_of' or 'related_to'."
    )
    try:
        # Use the v1+ OpenAI interface
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        triples = json.loads(text)
    except json.JSONDecodeError:
        st.error(f"Model returned non-JSON for '{concept}':\n{text}")
        return []
    except Exception as e:
        st.error(f"OpenAI API error for '{concept}': {e}")
        return []
    return triples


def build_graph(seed: str, depth: int = 2, max_triples: int = 6):
    G = nx.DiGraph()
    seen = {seed}
    queue = deque([(seed, 0)])
    while queue:
        node, level = queue.popleft()
        if level >= depth:
            continue
        triples = extract_triples(node, max_triples)
        for subject, relation, obj in triples:
            G.add_node(subject)
            G.add_node(obj)
            G.add_edge(subject, obj, relation=relation)
            if obj not in seen:
                seen.add(obj)
                queue.append((obj, level + 1))
    return G


def detect_communities(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return {}
    return community_louvain.best_partition(G.to_undirected())


def visualize_graph(G: nx.DiGraph, partition: dict):
    net = Network(height="700px", width="100%", directed=True)
    for node in G.nodes():
        cid = partition.get(node, 0)
        color = f"hsl({(cid*60)%360},70%,50%)"
        net.add_node(node, label=node, color=color)
    for u, v, data in G.edges(data=True):
        style = {'dashes': data['relation']!='subclass_of'}
        net.add_edge(u, v, title=data['relation'], **style)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.show(path)
    return open(path, 'r', encoding='utf-8').read()


def main():
    st.title("🔍 Knowledge Graph Explorer")
    seed = st.text_input("Seed topic", "Spatial data warehouse")
    depth = st.slider("Depth", 1, 3, 2)
    max_triples = st.slider("Triples/node", 1, 8, 4)

    if st.button("Generate Graph"):
        with st.spinner("Building graph..."):
            G = build_graph(seed, depth, max_triples)
            if not G.nodes:
                st.warning("No data; try a different seed or settings.")
                return
            partition = detect_communities(G)
            html = visualize_graph(G, partition)
            st.components.v1.html(html, height=700, scrolling=True)

if __name__ == "__main__":
    main()
