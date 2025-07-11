import streamlit as st
import re
import networkx as nx
from pyvis.network import Network
from openai import OpenAI
import json

# ─── CONFIG ────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def get_llm_neighbors(term: str, rel: str, limit: int) -> list[str]:
    if rel == "subtopic":
        prompt = f"Provide a JSON array of up to {limit} concise, distinct subtopics (more specific topics) of \"{term}\"."
    elif rel == "related":
        prompt = f"Provide a JSON array of up to {limit} concise, distinct concepts related to but not subtopics of \"{term}\"."
    elif rel == "related_question":
        prompt = f"Provide a JSON array of up to {limit} distinct user search queries (phrased as questions) related to \"{term}\"."
    else:
        return []

    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You output only a JSON array of strings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    content = resp.choices[0].message.content
    try:
        arr = json.loads(content)
        return [str(item) for item in arr][:limit]
    except json.JSONDecodeError:
        items = [re.sub(r"^[-•\s]+", "", line).strip() for line in content.splitlines() if line.strip()]
        return items[:limit]

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_full_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed", depth=0)

    level1 = get_llm_neighbors(seed, "subtopic", max_sub)
    for topic in level1:
        G.add_node(topic, label=topic, rel="subtopic", depth=1)
        G.add_edge(seed, topic)
    if sub_depth > 1:
        for topic in level1:
            level2 = get_llm_neighbors(topic, "subtopic", max(1, max_sub // 2))
            for sub2 in level2:
                if not G.has_node(sub2):
                    G.add_node(sub2, label=sub2, rel="subtopic", depth=2)
                G.add_edge(topic, sub2)

    related = get_llm_neighbors(seed, "related", max_rel)
    for concept in related:
        G.add_node(concept, label=concept, rel="related", depth=1)
        G.add_edge(seed, concept)
    for concept in related:
        subrel = get_llm_neighbors(concept, "related", sem_sub_lim)
        for sr in subrel:
            if not G.has_node(sr):
                G.add_node(sr, label=sr, rel="related", depth=2)
            G.add_edge(concept, sr)

    if include_q:
        questions = get_llm_neighbors(seed, "related_question", max_q)
        for q in questions:
            G.add_node(q, label=q, rel="related_question", depth=1)
            G.add_edge(seed, q)

    return G

def filter_graph(G, show_subtopics, show_related, show_questions):
    H = nx.Graph()
    for node, data in G.nodes(data=True):
        if data['rel'] == 'seed' or \
           (data['rel'] == 'subtopic' and show_subtopics) or \
           (data['rel'] == 'related' and show_related) or \
           (data['rel'] == 'related_question' and show_questions):
            H.add_node(node, **data)
    for u, v in G.edges():
        if H.has_node(u) and H.has_node(v):
            H.add_edge(u, v)
    return H

# ─── VISUALIZE ───────────────────────────────────────────
def draw_pyvis(G: nx.Graph):
    net = Network(height="750px", width="100%", notebook=False)
    net.set_options("""
    var options = {
      "interaction": {"hover": true, "navigationButtons": true},
      "physics": {"enabled": true, "stabilization": {"iterations": 300}}
    }
    """)
    colors = {"seed": "#1f78b4", "subtopic": "#66c2a5", "related": "#61b2ff", "related_question": "#ffcc61"}
    for node, data in G.nodes(data=True):
        net.add_node(node, label=data["label"], title=f"{data['rel']} (depth {data['depth']})", color=colors.get(data['rel'], "#999999"))
    for u, v in G.edges():
        net.add_edge(u, v)
    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 LLM-driven Knowledge Graph Generator")

with st.sidebar:
    st.header("Controls")
    seed = st.text_input("Seed topic", "data warehouse")
    max_sub = st.slider("Max subtopics", 5, 50, 20)
    max_rel = st.slider("Max related concepts", 5, 50, 20)
    include_q = st.checkbox("Include related questions", value=True)

    st.subheader("Toggle Display")
    show_subtopics = st.checkbox("Show Subtopics", True)
    show_related = st.checkbox("Show Related Concepts", True)
    show_questions = st.checkbox("Show Related Questions", True)

    sub_depth = 1
    max_q = 20
    sem_sub_lim = max_rel // 2

if st.sidebar.button("Generate Graph"):
    with st.spinner("Building graph…"):
        full_G = build_full_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q)
        G = filter_graph(full_G, show_subtopics, show_related, show_questions)
    st.success(f"✅ Nodes: {len(G.nodes)} Edges: {len(G.edges)}")
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
