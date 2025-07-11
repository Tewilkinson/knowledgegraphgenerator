import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI

# ──────────────────────────────────────────────────────────
# 1. LOOKUP: label → Wikidata QID
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str | None:
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label,
            "limit": 1
        },
        timeout=5
    )
    resp.raise_for_status()
    hits = resp.json().get("search", [])
    return hits[0]["id"] if hits else None

# ──────────────────────────────────────────────────────────
# 2. TAXONOMY: direct subclasses (“Subtopics”) via P279
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_subclasses(qid: str, limit: int = 20):
    sparql = f"""
    SELECT ?child ?childLabel WHERE {{
      ?child wdt:P279 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    r = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": sparql},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    )
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [
        (row["child"]["value"].rsplit("/",1)[-1],
         row["childLabel"]["value"])
        for row in rows
    ]

# ──────────────────────────────────────────────────────────
# 3. SEMANTIC: ConceptNet neighbours (“Related Entities”)
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit},
        timeout=5
    )
    r.raise_for_status()
    rels = r.json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0))
        for e in rels
    ]

# ──────────────────────────────────────────────────────────
# 4. GPT: seed → “related queries”
# ──────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data(show_spinner=False)
def get_gpt_neighbors(seed: str, limit: int = 20):
    prompt = (
        f"List {limit} concise, distinct search queries related to “{seed}”. "
        "Return them as a bulleted list with one per line, no numbering."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7
    )
    lines = resp.choices[0].message.content.splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        # strip bullets/dashes
        clean = ln.lstrip("-• ").rstrip()
        out.append(clean)
    return out

# ──────────────────────────────────────────────────────────
# 5. BUILD the hybrid graph
# ──────────────────────────────────────────────────────────
def build_graph(seed: str,
                depth: int,
                tax_limit: int,
                sem_limit: int,
                gpt_limit: int) -> nx.Graph:
    qid = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(seed, label=seed, rel="seed")

    # 1-hop Subtopics (Wikidata P279)
    if qid:
        for child_q, child_lbl in get_subclasses(qid, tax_limit):
            G.add_node(child_q, label=child_lbl, rel="subtopic")
            G.add_edge(seed, child_q)

    # 2-hop Subtopics
    if depth > 1 and qid:
        first = [n for n,d in G.nodes(data=True) if d["rel"]=="subtopic"]
        for n in first:
            for cq, cl in get_subclasses(n, max(tax_limit//2,1)):
                if not G.has_node(cq):
                    G.add_node(cq, label=cl, rel="subtopic")
                G.add_edge(n, cq)

    # 1-hop Related Entities (ConceptNet)
    for lbl, _ in get_conceptnet_neighbors(seed, sem_limit):
        nid = f"CN:{lbl}"
        G.add_node(nid, label=lbl, rel="related")
        G.add_edge(seed, nid)

    # 1-hop GPT queries
    for qry in get_gpt_neighbors(seed, gpt_limit):
        nid = f"GPT:{qry}"
        G.add_node(nid, label=qry, rel="gpt")
        G.add_edge(seed, nid)

    return G

# ──────────────────────────────────────────────────────────
# 6. STREAMLIT UI
# ──────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 Hybrid: Wikidata P279 + ConceptNet + GPT")

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Subtopic depth",          1, 3, 1)
tax_lim = st.slider("Max subtopics (P279)",    5, 50, 20)
sem_lim = st.slider("Max ConceptNet items",    5, 50, 20)
gpt_lim = st.slider("Max GPT queries",         5, 50, 20)

if st.button("Generate Graph"):
    G   = build_graph(seed, depth, tax_lim, sem_lim, gpt_lim)
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # edges
    ex, ey = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]

    edge_trace = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="#888",width=1), hoverinfo="none"
    )

    # nodes by type
    traces = []
    color_map = {
        "seed":    "#ff6961",
        "subtopic":"#61ff8e",
        "related": "#61b2ff",
        "gpt":     "#ffcc61"
    }
    for rel_type, color in color_map.items():
        xs, ys, txt = [], [], []
        for n,d in G.nodes(data=True):
            if d["rel"] == rel_type:
                x,y = pos[n]
                xs.append(x); ys.append(y); txt.append(d["label"])
        traces.append(
            go.Scatter(
                x=xs, y=ys, mode="markers+text",
                text=txt, textposition="top center",
                marker=dict(size=18, color=color),
                name=rel_type.capitalize(), hoverinfo="text"
            )
        )

    fig = go.Figure(data=[edge_trace] + traces)
    fig.update_layout(
        showlegend=True, margin=dict(l=20,r=20,t=40,b=20),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)
