import streamlit as st
import re
import networkx as nx
from pyvis.network import Network
from openai import OpenAI
import json
import pandas as pd
from pytrends.request import TrendReq

# ─── CONFIG ────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pytrends = TrendReq(hl='en-US', tz=360)

# ─── STREAMLIT PAGE SETUP ──────────────────────────────
st.set_page_config(layout="wide")
# Inject CSS to force full-viewport width
st.markdown(
    """
    <style>
      .stApp .css-1d391kg, .stApp [data-testid="stAppViewContainer"] {
        max-width: 100vw !important;
        width: 100vw !important;
        padding: 0 !important;
      }
      iframe {
        width: 100% !important;
        margin: 0;
      }
      [data-testid="stVerticalBlock"] > [data-testid="stHtmlBlock"] > div {
        width: 100vw !important;
        max-width: 100vw !important;
        padding: 0 !important;
        margin: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def get_llm_neighbors(term: str, rel: str, limit: int) -> list[str]:
    if rel == "subtopic":
        prompt = (
            f"Provide a JSON array of up to {limit} concise, distinct subtopics "
            f"of '{term}'."
        )
    elif rel == "related":
        prompt = (
            f"Provide a JSON array of up to {limit} concise, distinct concepts "
            f"related to but not subtopics of '{term}'."
        )
    elif rel == "related_question":
        prompt = (
            f"Provide a JSON array of up to {limit} distinct user search queries "
            f"(as questions) related to '{term}'."
        )
    else:
        return []
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Output only a JSON array of strings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    content = resp.choices[0].message.content
    try:
        arr = json.loads(content)
        return [str(i) for i in arr][:limit]
    except json.JSONDecodeError:
        lines = [re.sub(r"^[-•\s]+", "", l).strip() for l in content.splitlines()]
        return [l for l in lines if l][:limit]

@st.cache_data
def classify_topic(topic: str, seed: str) -> dict:
    prompt = (
        f"Determine whether '{topic}' is a subtopic of '{seed}', or a standalone seed-level topic. "
        f"Respond only with JSON: {{'relation': 'subtopic' or 'seed', 'parent': <if subtopic then seed else null>}}."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {'relation': 'seed', 'parent': None}

# ─── GRAPH FUNCTIONALITY ─────────────────────────────────
def build_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed", depth=0)
    # subtopics
    lvl1 = get_llm_neighbors(seed, "subtopic", max_sub)
    for t in lvl1:
        G.add_node(t, label=t, rel="subtopic", depth=1)
        G.add_edge(seed, t)
    if sub_depth > 1:
        for t in lvl1:
            lvl2 = get_llm_neighbors(t, "subtopic", max(1, max_sub//2))
            for t2 in lvl2:
                if not G.has_node(t2): G.add_node(t2, label=t2, rel="subtopic", depth=2)
                G.add_edge(t, t2)
    # related
    rel = get_llm_neighbors(seed, "related", max_rel)
    for c in rel:
        G.add_node(c, label=c, rel="related", depth=1)
        G.add_edge(seed, c)
    for c in rel:
        subr = get_llm_neighbors(c, "related", sem_sub_lim)
        for sr in subr:
            if not G.has_node(sr): G.add_node(sr, label=sr, rel="related", depth=2)
            G.add_edge(c, sr)
    # questions
    if include_q:
        qs = get_llm_neighbors(seed, "related_question", max_q)
        for q in qs:
            G.add_node(q, label=q, rel="related_question", depth=1)
            G.add_edge(seed, q)
    # trends
    for n in G.nodes:
        tr = get_topic_trend(G.nodes[n]['label']) if 'get_topic_trend' in globals() else None
        if tr:
            G.nodes[n].update(tr)
    return G

# ─── VISUALIZE PYVIS ─────────────────────────────────────
def draw_pyvis(G):
    net = Network(height="750px", width="100%", notebook=False)
    net.set_options("""
var options={interaction:{hover:true,navigationButtons:true},physics:{stabilization:{iterations:300}}}
""")
    cols={"seed":"#1f78b4","subtopic":"#66c2a5","related":"#61b2ff","related_question":"#ffcc61"}
    for node,data in G.nodes(data=True):
        title=f"{data['rel']} (depth {data['depth']})"
        if 'volume' in data:
            title+=f"<br>🔍{data['volume']}📈{data['yoy']}%"
        net.add_node(node,label=data['label'],title=title,color=cols.get(data['rel']))
    for u,v in G.edges(): net.add_edge(u,v)
    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.title("LLM-driven Knowledge Graph & Topic Checker")

tab1, tab2 = st.tabs(["Knowledge Graph","Topic Structure Checker"])

with tab1:
    with st.sidebar:
        st.header("Graph Controls")
        seed = st.text_input("Seed topic","data warehouse")
        max_sub = st.slider("Max subtopics",5,50,20)
        max_rel = st.slider("Max related concepts",5,50,20)
        include_q = st.checkbox("Include Qs",True)
        show_adv = st.checkbox("Advanced")
        if show_adv:
            sub_depth=st.slider("Levels",1,2,1)
            max_q=st.slider("# Qs",5,50,20)
            sem_sub_lim=st.slider("2nd-level rel",0,max_rel,max_rel//2)
        else:
            sub_depth, max_q, sem_sub_lim = 1,20,max_rel//2
    if st.sidebar.button("Generate Graph"):
        G=build_graph(seed,sub_depth,max_sub,max_rel,sem_sub_lim,include_q,max_q)
        st.success(f"Nodes:{len(G.nodes)} Edges:{len(G.edges)}")
        html=draw_pyvis(G)
        st.components.v1.html(html,height=800,scrolling=True,width=2000)
        # CSV export
        df=pd.DataFrame([{
            'Topic':d['label'],'Type':d['rel'],'Depth':d['depth'],
            'Volume':d.get('volume',''),'YoY':d.get('yoy',''),'Trend':d.get('3mo_trend','')
        } for _,d in G.nodes(data=True)])
        st.download_button("Download CSV",df.to_csv(index=False),"graph.csv","text/csv")

with tab2:
    st.header("Check Topic Structure")
    seed_check = st.text_input("Existing seed topic","data warehouse",key="s2")
    topic_check = st.text_input("Topic to classify","etl process",key="t2")
    if st.button("Check Structure",key="chk"):
        with st.spinner("Classifying..."):
            res = classify_topic(topic_check.lower(),seed_check.lower())
        if res.get('relation')=='subtopic':
            st.success(f"‘{topic_check}’ is a subtopic of ‘{res.get('parent',seed_check)}’.")
        else:
            st.info(f"‘{topic_check}’ appears to be a standalone seed-level topic.")
