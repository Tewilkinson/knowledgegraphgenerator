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
# community detection if desired
import community as community_louvain  # pip install python-louvain
from pyvis.network import Network  # pip install pyvis

# Page configuration
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_openai(prompt: str) -> list[str]:
    """
    Calls OpenAI and expects a JSON array of strings.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        st.error(f"OpenAI error or parse failure:\n{text if 'text' in locals() else ''}\n{e}")
        return []


def extract_relations(concept: str, relation: str, description: str, max_count: int):
    """
    Extracts related concepts of type 'description' for a given concept.
    Returns list of related concept names.
    """
    prompt = (
        f"You are a knowledge graph extractor. List up to {max_count} {description} "
        f"of '{concept}' as a JSON array of strings."
    )
    return call_openai(prompt)


def build_graph(seed: str, depth: int = 2, max_count: int = 5):
    """
    Builds a directed graph with two edge types: 'subclass_of' (solid) and 'related_to' (dashed).
    Only subtopics ('subclass_of') are recursively expanded up to 'depth'.
    """
    G = nx.DiGraph()
    seen = set([seed])
    queue = deque([(seed, 0)])
    G.add_node(seed)

    while queue:
        node, level = queue.popleft()
        if level >= depth:
            continue
        # 1) subtopics (hierarchical)
        subs = extract_relations(node, "subclass_of", "subtopics", max_count)
        for sub in subs:
            G.add_node(sub)
            G.add_edge(node, sub, relation="subclass_of")
            if sub not in seen:
                seen.add(sub)
                queue.append((sub, level + 1))
        # 2) related entities (non-hierarchical)
        rels = extract_relations(node, "related_to", "related entities", max_count)
        for rel in rels:
            G.add_node(rel)
            G.add_edge(node, rel, relation="related_to")
    return G


def visualize_graph(G: nx.DiGraph):
    """
    Visualizes graph hierarchically with solid edges for 'subclass_of' and dashed for 'related_to'.
    """
    net = Network(height="700px", width="100%", directed=True)
    # enable hierarchical layout
    net.set_options(
        """
        var options = {
          layout: { hierarchical: { enabled: true, direction: 'LR', sortMethod: 'hubsize' } },
          edges: { smooth: true }
        }
        """
    )
    for node in G.nodes():
        # color by whether it's root, subtopic, or related (community optional)
        net.add_node(node, label=node)
    for u, v, data in G.edges(data=True):
        style = {'dashes': data['relation'] != 'subclass_of'}
        net.add_edge(u, v, title=data['relation'], **style)

    # write html
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.show(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    st.title("🔍 Knowledge Graph Explorer")
    seed = st.text_input("Seed topic", "Spatial data warehouse")
    depth = st.slider("Subtopic depth", 1, 3, 2)
    max_rel = st.slider("Max items per relation", 1, 8, 5)

    if st.button("Generate Knowledge Graph"):
        with st.spinner("Building graph…"):
            G = build_graph(seed, depth, max_rel)
            if G.number_of_nodes() <= 1:
                st.warning("No related concepts found. Try a different seed or increase limits.")
            else:
                html = visualize_graph(G)
                st.components.v1.html(html, height=700, scrolling=True)

if __name__ == '__main__':
    main()
