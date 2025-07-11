import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI
import re

# ───────────────────────────────────────────────────────────
# 1. WIKIDATA LOOKUP
# ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str | None:
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action":"wbsearchentities",
                "format":"json",
                "language":"en",
                "search":label,
                "limit":1
            },
            timeout=5
        )
        r.raise_for_status()
        hits = r.json().get("search",[])
        return hits[0]["id"] if hits else None
    except:
        return None

# ───────────────────────────────────────────────────────────
# 2. WIKIDATA P279 SUBTOPICS
# ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_subclasses(qid: str, limit: int=20):
    sparql = f"""
    SELECT ?child ?childLabel WHERE {{
      ?child wdt:P279 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    r = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query":sparql},
        headers={"Accept":"application/sparql-results+json"},
        timeout=10
    )
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [
        (row["child"]["value"].rsplit("/",1)[-1],
         row["childLabel"]["value"])
        for row in rows
    ]

# ───────────────────────────────────────────────────────────
# 3. CONCEPTNET RELATED
# ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int=20):
    uri = concept.lower().replace(" ","_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit},
        timeout=5
    )
    r.raise_for_status()
    return [
        e["@id"].split("/")[-1].replace("_"," ")
        for e in r.json().get("related",[])
    ]

# ───────────────────────────────────────────────────────────
# 4. GPT “RELATED QUERIES”
# ───────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data(show_spinner=False)
def get_gpt_neighbors(seed: str, limit: int=20):
    prompt = (
        f"List {limit} concise, distinct search queries related to “{seed}”. "
        "Return as a bulleted list, one per line, no numbering."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7,
    )
    out = []
    for ln in resp.choices[0].message.content.splitlines():
        ln = ln.strip()
        if not ln: continue
        clean = re.sub(r"^[-•\s]+","", ln)
        out.append(clean)
    return out

# ───────────────────────────────────────────────────────────
# 5. BUILD HYBRID GRAPH
# ───────────────────────────────────────────────────────────
def build_graph(seed, sub_depth, tax_lim, sem_lim, gpt_seed_lim, gpt_rel_lim):
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed")

    # 1-hop P279 subtopics
    qid = lookup_qid(seed)
    if qid:
        lvl1 = get_subclasses(qid, tax_lim)
        for cid, clbl in lvl1:
            G.add_node(clbl, label=clbl, rel="subtopic")
            G.add_edge(seed, clbl)
        # optional 2-hop
        if sub_depth>1:
            for cid, clbl in lvl1:
                cqid = lookup_qid(clbl)
                if cqid:
                    for c2id, c2lbl in get_subclasses(cqid, max(1,tax_lim//2)):
                        if not G.has_node(c2lbl):
                            G.add_node(c2lbl, label=c2lbl, rel="subtopic")
                        G.add_edge(clbl, c2lbl)

    # 1-hop ConceptNet related
    sems = get_conceptnet_neighbors(seed, sem_lim)
    for lbl in sems:
        G.add_node(lbl, label=lbl, rel="related")
        G.add_edge(seed, lbl)

    # 1-hop GPT on seed
    gpts = get_gpt_neighbors(seed, gpt_seed_lim)
    for qry in gpts:
        G.add_node(qry, label=qry, rel="gpt_seed")
        G.add_edge(seed, qry)

    # 1-hop GPT on each ConceptNet related node
    for lbl in sems:
        subs = get_gpt_neighbors(lbl, gpt_rel_lim)
        for sub in subs:
            G.add_node(sub, label=sub, rel="gpt_related")
            G.add_edge(lbl, sub)

    return G

# ───────────────────────────────────────────────────────────
# 6. STREAMLIT UI & RENDER
# ───────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 Hybrid + GPT on Related Topics")

seed     = st.text_input("Seed topic",             "data warehouse")
sub_d    = st.slider("Subtopic P279 depth",       1, 2, 1)
tax_lim  = st.slider("Max P279 subtopics",        5, 50, 20)
sem_lim  = st.slider("Max ConceptNet related",    5, 50, 20)
gpt_seed = st.slider("GPT queries on seed",       5, 50, 20)
gpt_rel  = st.slider("GPT queries on related nodes", 2, 20, 5)

if st.button("Generate Hybrid Graph"):
    G = build_graph(seed, sub_d, tax_lim, sem_lim, gpt_seed, gpt_rel)
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # edge trace
    ex, ey = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(color="#888",width=1), hoverinfo="none")

    # node traces by rel type
    color_map = {
        "seed":        "#ff6961",
        "subtopic":    "#61ff8e",
        "related":     "#61b2ff",
        "gpt_seed":    "#ffcc61",
        "gpt_related": "#ff61cc"
    }
    traces = []
    for rel, color in color_map.items():
        xs, ys, txt = [], [], []
        for n,d in G.nodes(data=True):
            if d["rel"]==rel:
                x,y = pos[n]
                xs.append(x); ys.append(y); txt.append(d["label"])
        traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=txt, textposition="top center",
                marker=dict(size=16, color=color),
                name=rel.replace("_"," ").title(),
                hoverinfo="text"
            )
        )

    fig = go.Figure(data=[edge_trace]+traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20,r=20,t=40,b=20),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)
