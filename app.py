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
    """
    Fetch neighbors of a given type using ChatGPT:
      - rel='subtopic': more specific topics (subclasses)
      - rel='related': related concepts
      - rel='related_question': user search questions
    Returns a list of strings.
    """
    if rel == "subtopic":
        prompt = (
            f"Provide a JSON array of up to {limit} concise, distinct subtopics "
            f"(more specific topics) of \"{term}\"."
        )
    elif rel == "related":
        prompt = (
            f"Provide a JSON array of up to {limit} concise, distinct concepts "
            f"related to but not subtopics of \"{term}\"."
        )
    elif rel == "related_question":
        prompt = (
            f"Provide a JSON array of up to {limit} distinct user search "
            f"queries (phrased as questions) related to \"{term}\"."
        )
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
        items = []
        for line in content.splitlines():
            clean = re.sub(r"^[-•\s]+", "", line).strip()
            if clean:
                items.append(clean)
        return items[:limit]

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # Subtopics (level 1 & optional level 2)
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

    # Related concepts
    related = get_llm_neighbors(seed, "related", max_rel)
    for concept in related:
        G.add_node(concept, label=concept, rel="related", depth=1)
        G.add_edge(seed, concept)
    # Second-level related
    for concept in related:
        subrel = get_llm_neighbors(concept, "related", sem_sub_lim)
        for sr in subrel:
            if not G.has_node(sr):
                G.add_node(sr, label=sr, rel="related", depth=2)
            G.add_edge(concept, sr)

    # Related questions
    if include_q:
        questions = get_llm_neighbors(seed, "related_question", max_q)
        for q in questions:
            G.add_node(q, label=q, rel="related_question", depth=1)
            G.add_edge(seed, q)
    return G

# ─── VISUALIZE ───────────────────────────────────────────
def draw_pyvis(G: nx.Graph):
    net = Network(height="750px", width="100%", notebook=False)
    net.set_options("""
var options = {
  "interaction": {"hover": true, "navigationButtons": true},
  "physics": {"enabled": true, "stabilization": {"iterations": 300}}
}
""")
    # Uniform circle shapes, fixed size
    props = {
        "seed": {"color": "red"},
        "subtopic": {"color": "#66c2a5"},
        "related": {"color": "#61b2ff"},
        "related_question": {"color": "#61b2ff"}
    }
    for node, data in G.nodes(data=True):
        p = props.get(data["rel"], {"color": "#999999"})
        net.add_node(
            node,
            label=data["label"],
            title=f"{data['rel']} (depth {data['depth']})",
            color=p["color"],
            shape="circle",
            size=25
        )
    for u, v in G.edges():
        net.add_edge(u, v)
    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 LLM-driven Knowledge Graph Generator")

# Sidebar
with st.sidebar:
    st.header("Controls")
    seed = st.text_input("Seed topic", "data warehouse")
    max_sub = st.slider("Max subtopics", 5, 50, 20)
    max_rel = st.slider("Max related concepts", 5, 50, 20)
    include_q = st.checkbox("Include related questions", value=True)
    show_adv = st.checkbox("Show advanced settings")
    if show_adv:
        sub_depth = st.slider("Subtopic depth (levels)", 1, 2, 1)
        max_q = st.slider("Number of questions", 5, 50, 20)
        sem_sub_lim = st.slider("Max second-level related", 0, max_rel, max_rel // 2)
    else:
        sub_depth = 1
        max_q = 20
        sem_sub_lim = max_rel // 2

if st.sidebar.button("Generate Graph"):
    with st.spinner("Building graph…"):
        G = build_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    st.markdown(
        "<span style='color:red;'>🔴</span> Seed "
        "<span style='color:#66c2a5;'>🔵</span> Subtopic  "
        "<span style='color:#61b2ff;'>🔵</span> Related & Questions",
        unsafe_allow_html=True
    )
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
