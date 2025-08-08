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
st.markdown(
    '''
    <style>
      /* Full-viewport width */
      .stApp .css-1d391kg, .stApp [data-testid="stAppViewContainer"] {
        width: 100vw !important; max-width: 100vw !important; padding: 0 !important;
      }
      iframe { width: 100% !important; margin: 0; }
      [data-testid="stVerticalBlock"] > [data-testid="stHtmlBlock"] > div {
        width: 100vw !important; max-width: 100vw !important; padding: 0; margin: 0;
      }
    </style>
    ''', unsafe_allow_html=True
)

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def get_llm_neighbors(term: str, rel: str, limit: int) -> list[str]:
    if rel == "subtopic":
        prompt = f"Provide a JSON array of up to {limit} concise, distinct subtopics of '{term}'."
    elif rel == "related":
        prompt = f"Provide a JSON array of up to {limit} concise, distinct concepts related to but not subtopics of '{term}'."
    elif rel == "related_question":
        prompt = f"Provide a JSON array of up to {limit} distinct user search queries (as questions) related to '{term}'."
    else:
        return []
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Output only a JSON array of strings."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = resp.choices[0].message.content
    try:
        arr = json.loads(content)
        return [str(i) for i in arr][:limit]
    except json.JSONDecodeError:
        return [re.sub(r"^[-•\s]+", "", l).strip() for l in content.splitlines() if l][:limit]

@st.cache_data
def find_parent_topic_weights(topic: str, candidates: list[str]) -> pd.DataFrame:
    """
    Given a topic and list of candidate parent topics, ask the LLM to score each candidate
    by relevance 0-100. Returns a DataFrame with columns ['parent','score'].
    """
    prompt = (
        f"For the topic '{topic}', assign a relevance score from 0 to 100 to each of the following higher-level domains: "
        f"{', '.join(candidates)}. Respond only as JSON array of objects with 'parent' and 'score' fields."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Output only valid JSON."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        arr = json.loads(resp.choices[0].message.content)
        df = pd.DataFrame(arr)
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
        return df.sort_values('score', ascending=False).reset_index(drop=True)
    except Exception:
        # fallback: equal weighting
        df = pd.DataFrame({'parent': candidates, 'score': [100//len(candidates)]*len(candidates)})
        return df.sort_values('score', ascending=False).reset_index(drop=True)

# ─── GRAPH & VISUALS ───────────────────────────────────
def build_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q):
    G = nx.Graph(); G.add_node(seed, label=seed, rel="seed", depth=0)
    for t in get_llm_neighbors(seed, "subtopic", max_sub): G.add_node(t,label=t,rel="subtopic",depth=1); G.add_edge(seed,t)
    if sub_depth>1:
        for t in list(G.nodes):
            if G.nodes[t]['rel']=='subtopic':
                for t2 in get_llm_neighbors(t, "subtopic", max(1, max_sub//2)):
                    if not G.has_node(t2): G.add_node(t2,label=t2,rel="subtopic",depth=G.nodes[t]['depth']+1)
                    G.add_edge(t,t2)
    for c in get_llm_neighbors(seed, "related", max_rel):
        G.add_node(c,label=c,rel="related",depth=1); G.add_edge(seed,c)
        for sr in get_llm_neighbors(c, "related", sem_sub_lim):
            if not G.has_node(sr): G.add_node(sr,label=sr,rel="related",depth=2)
            G.add_edge(c,sr)
    if include_q:
        for q in get_llm_neighbors(seed, "related_question", max_q): G.add_node(q,label=q,rel="related_question",depth=1); G.add_edge(seed,q)
    return G

def draw_pyvis(G: nx.Graph) -> str:
    net = Network(height="750px", width="100%", notebook=False)
    net.set_options("""var options={interaction:{hover:true,navigationButtons:true},physics:{stabilization:{iterations:300}}}""")
    cols={"seed":"#1f78b4","subtopic":"#66c2a5","related":"#61b2ff","related_question":"#ffcc61"}
    for n,d in G.nodes(data=True): net.add_node(n,label=d['label'],title=f"{d['rel']} (depth {d['depth']})",color=cols.get(d['rel'],"#999"))
    for u,v in G.edges(): net.add_edge(u,v)
    return net.generate_html()

# ─── APP UI ─────────────────────────────────────────────
st.title("LLM-driven Knowledge Graph & Parent Topic Weigher")

tab1, tab2 = st.tabs(["Knowledge Graph","Parent Topic Weigher"])

with tab1:
    with st.sidebar:
        st.header("Graph Controls")
        seed = st.text_input("Seed topic","data warehouse")
        max_sub = st.slider("Max subtopics",5,50,20)
        max_rel = st.slider("Max related",5,50,20)
        include_q = st.checkbox("Include questions",True)
        show_adv = st.checkbox("Advanced settings")
        if show_adv:
            sub_depth = st.slider("Subtopic depth",1,2,1)
            max_q = st.slider("# questions",5,50,20)
            sem_sub_lim = st.slider("2nd level related",0,max_rel,max_rel//2)
        else:
            sub_depth, max_q, sem_sub_lim = 1,20,max_rel//2
    if st.sidebar.button("Generate Graph"):
        G = build_graph(seed, sub_depth, max_sub, max_rel, sem_sub_lim, include_q, max_q)
        st.success(f"Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
        html = draw_pyvis(G)
        st.components.v1.html(html, height=800, scrolling=True, width=2000)
        df = pd.DataFrame([{'Topic':d['label'],'Type':d['rel'],'Depth':d['depth']} for _,d in G.nodes(data=True)])
        st.download_button("Download CSV", df.to_csv(index=False), "graph.csv", "text/csv")

with tab2:
    st.header("Parent Topic Weigher")
    topic = st.text_input("Topic to weigh","etl process")
    candidates = st.text_area(
        "Candidate parents (comma-separated)",
        value="Data Engineering, Data Integration, Data Warehousing, Business Intelligence, Data Management"
    ).split(",")
    candidates = [c.strip() for c in candidates if c.strip()]
    if st.button("Compute Weights"):
        with st.spinner("Scoring..."):
            weights_df = find_parent_topic_weights(topic, candidates)
        st.bar_chart(weights_df.set_index('parent')['score'])
        top = weights_df.iloc[0]
        st.markdown(f"**Top fit:** '{topic}' is most strongly a subtopic of **{top['parent']}** (score {top['score']}).")
        st.dataframe(weights_df)
