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
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON arrays of strings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    content = resp.choices[0].message.content
    try:
        arr = json.loads(content)
        # ensure list of strings
        return [str(item) for item in arr][:limit]
    except json.JSONDecodeError:
        # fallback: parse lines
        out = []
        for ln in content.splitlines():
            clean = re.sub(r"^[-•\s]+", "", ln.strip())
            if clean:
                out.append(clean)
        return out[:limit]

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_graph(seed, sub_depth, tax_lim, sem_lim, sem_sub_lim, rq_seed_lim):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # ChatGPT-derived subtopics
    lvl1 = get_llm_neighbors(seed, "subtopic", tax_lim)
    for clbl in lvl1:
        G.add_node(clbl, label=clbl, rel="subtopic", depth=1)
        G.add_edge(seed, clbl)
    if sub_depth > 1:
        for clbl in lvl1:
            lvl2 = get_llm_neighbors(clbl, "subtopic", max(1, tax_lim // 2))
            for c2lbl in lvl2:
                if not G.has_node(c2lbl):
                    G.add_node(c2lbl, label=c2lbl, rel="subtopic", depth=2)
                G.add_edge(clbl, c2lbl)

    # ChatGPT-derived related concepts
    sems = get_llm_neighbors(seed, "related", sem_lim)
    for lbl in sems:
        G.add_node(lbl, label=lbl, rel="related", depth=1)
        G.add_edge(seed, lbl)

    # Second-level related
    for lbl in sems:
        secs = get_llm_neighbors(lbl, "related", sem_sub_lim)
        for sl in secs:
            if not G.has_node(sl):
                G.add_node(sl, label=sl, rel="related", depth=2)
            G.add_edge(lbl, sl)

    # ChatGPT-derived related questions
    rqs = get_llm_neighbors(seed, "related_question", rq_seed_lim)
    for qry in rqs:
        G.add_node(qry, label=qry, rel="related_question", depth=1)
        G.add_edge(seed, qry)

    return G

# ─── VISUALIZE ───────────────────────────────────────────
def draw_pyvis(G: nx.Graph):
    net = Network(height="750px", width="100%", notebook=False)
    net.set_options("""
    var options = {
      "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
      "physics": {"enabled": true, "stabilization": {"iterations": 500}}
    }
    """)

    color_map = {
        "seed": "#1f78b4",
        "subtopic": "#66c2a5",
        "related": "#61b2ff",
        "related_question": "#ffcc61",
    }
    for nid, data in G.nodes(data=True):
        net.add_node(
            nid,
            label=data["label"],
            title=f"{data['rel']} (depth {data['depth']})",
            color=color_map.get(data["rel"], "#999999")
        )
    for u, v in G.edges():
        net.add_edge(u, v)

    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 LLM-driven Knowledge Graph Generator")

# Sidebar controls
st.sidebar.header("Controls")
seed = st.sidebar.text_input("Seed topic", "data warehouse")
sub_d = st.sidebar.slider("Subtopic depth", 1, 2, 1)
tax_lim = st.sidebar.slider("Max subtopics", 5, 50, 20)
sem_lim = st.sidebar.slider("Max related terms", 5, 50, 20)
sem_sub_lim = st.sidebar.slider("Max sub-related terms", 0, 50, 10)
rq_seed = st.sidebar.slider("Related Questions", 5, 50, 20)

if st.sidebar.button("Generate Graph"):
    with st.spinner("Building graph…"):
        G = build_graph(seed, sub_d, tax_lim, sem_lim, sem_sub_lim, rq_seed)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    # Legend
    st.markdown(
        "<span style='color:#1f78b4;'>🔵</span>Seed  "
        "<span style='color:#66c2a5;'>🟢</span>Subtopic  "
        "<span style='color:#61b2ff;'>🔷</span>Related  "
        "<span style='color:#ffcc61;'>🟠</span>Related Questions",
        unsafe_allow_html=True
    )
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
