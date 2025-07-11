import streamlit as st
import re
import json
import networkx as nx
from pyvis.network import Network
from openai import OpenAI
import tempfile
import os

# ─── CONFIG ────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def get_llm_neighbors(term: str, rel: str, limit: int) -> list[str]:
    prompt = ""
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

# ─── BUILD NETWORK ──────────────────────────────────────
def build_pyvis_graph(seed, max_sub, max_rel, include_q, max_q, show_subtopics, show_related, show_questions):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed")

    if show_subtopics:
        subtopics = get_llm_neighbors(seed, "subtopic", max_sub)
        for topic in subtopics:
            G.add_node(topic, label=topic, rel="subtopic")
            G.add_edge(seed, topic)

    if show_related:
        related = get_llm_neighbors(seed, "related", max_rel)
        for concept in related:
            G.add_node(concept, label=concept, rel="related")
            G.add_edge(seed, concept)

    if include_q and show_questions:
        questions = get_llm_neighbors(seed, "related_question", max_q)
        for q in questions:
            G.add_node(q, label=q, rel="question")
            G.add_edge(seed, q)

    nt = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

    # Visual settings
    rel_color = {
        "seed": "green",
        "subtopic": "blue",
        "related": "orange",
        "question": "red"
    }

    for node, data in G.nodes(data=True):
        nt.add_node(node, label=data["label"], color=rel_color.get(data["rel"], "gray"), title=data["label"], group=data["rel"])

    for source, target in G.edges():
        nt.add_edge(source, target)

    nt.repulsion(node_distance=150, spring_length=200)
    return nt

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📎 Interactive LLM-driven Knowledge Graph")

with st.sidebar:
    st.header("Controls")
    seed = st.text_input("Seed topic", "data warehouse")
    max_sub = st.slider("Max subtopics", 5, 50, 20)
    max_rel = st.slider("Max related concepts", 5, 50, 20)
    include_q = st.checkbox("Include related questions", value=True)
    max_q = st.slider("Max questions", 5, 30, 10)

    st.subheader("Display Options")
    show_subtopics = st.checkbox("Show Subtopics", True)
    show_related = st.checkbox("Show Related Concepts", True)
    show_questions = st.checkbox("Show Related Questions", True)

if st.sidebar.button("Generate Graph"):
    with st.spinner("Building interactive graph..."):
        nt = build_pyvis_graph(seed, max_sub, max_rel, include_q, max_q, show_subtopics, show_related, show_questions)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            nt.save_graph(tmpfile.name)
            st.components.v1.html(open(tmpfile.name, 'r', encoding='utf-8').read(), height=800, scrolling=True)
        os.unlink(tmpfile.name)
