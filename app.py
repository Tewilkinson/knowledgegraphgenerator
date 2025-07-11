import streamlit as st
import re
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI
import json

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

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_full_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed", depth=0)

    level1 = get_llm_neighbors(seed, "subtopic", max_sub)
    for topic in level1:
        G.add_node(topic, label=topic, rel="subtopic", depth=1)
        G.add_edge(seed, topic)

    related = get_llm_neighbors(seed, "related", max_rel)
    for concept in related:
        G.add_node(concept, label=concept, rel="related", depth=1)
        G.add_edge(seed, concept)

    if include_q:
        questions = get_llm_neighbors(seed, "related_question", max_q)
        for q in questions:
            G.add_node(q, label=q, rel="related_question", depth=1)
            G.add_edge(seed, q)

    return G

# ─── FORCE-DIRECTED GRAPH WITH PLOTLY ────────────────────
def draw_plotly_force_directed(G, show_subtopics, show_related, show_questions):
    pos = nx.spring_layout(G, k=0.3, iterations=50)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=[],
            size=20,
            line_width=2
        )
    )

    color_map = {"seed": "#34a853", "subtopic": "#4285f4", "related": "#fbbc05", "related_question": "#ea4335"}

    for node, data in G.nodes(data=True):
        if (data['rel'] == 'seed' or
            (data['rel'] == 'subtopic' and show_subtopics) or
            (data['rel'] == 'related' and show_related) or
            (data['rel'] == 'related_question' and show_questions)):
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([data['label']])
            node_trace['marker']['color'] += tuple([color_map.get(data['rel'], "#999")])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Force-directed Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

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

if st.sidebar.button("Generate Graph"):
    with st.spinner("Building graph…"):
        full_G = build_full_graph(seed, 1, max_sub, max_rel, max_rel // 2, include_q, 20)
    fig = draw_plotly_force_directed(full_G, show_subtopics, show_related, show_questions)
    st.plotly_chart(fig, use_container_width=True)
