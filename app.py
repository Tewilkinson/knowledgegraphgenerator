import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go

# ——————————————————————————————————————————————————————————————————————————————
# 1. LOOKUP: label → Wikidata QID
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    r = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label
        },
    )
    r.raise_for_status()
    return r.json()["search"][0]["id"]

# ——————————————————————————————————————————————————————————————————————————————
# 2. TAXONOMY: direct subclasses (“Subtopics”) via P279
# ——————————————————————————————————————————————————————————————————————————————
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
        headers={"Accept": "application/sparql-results+json"}
    )
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [
        (row["child"]["value"].rsplit("/",1)[-1], row["childLabel"]["value"])
        for row in rows
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 3. SEMANTIC: related neighbours (“Related Entities”) from ConceptNet
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}
    )
    r.raise_for_status()
    rels = r.json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0))
        for e in rels
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 4. BUILD the hybrid graph
# ——————————————————————————————————————————————————————————————————————————————
def build_graph(seed: str, depth: int, tax_limit: int, sem_limit: int) -> nx.Graph:
    qid = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(qid, label=seed, rel="seed")

    # 1-hop Subtopics
    for child_q, child_lbl in get_subclasses(qid, tax_limit):
        G.add_node(child_q, label=child_lbl, rel="subtopic")
        G.add_edge(qid, child_q)

    # optional 2-hop Subtopics
    if depth > 1:
        first_level = [n for n,d in G.nodes(data=True) if d["rel"]=="subtopic"]
        for n in first_level:
            for cq, cl in get_subclasses(n, tax_limit//2):
                if not G.has_node(cq):
                    G.add_node(cq, label=cl, rel="subtopic")
                G.add_edge(n, cq)

    # 1-hop Related Entities
    for lbl, _ in get_conceptnet_neighbors(seed, sem_limit):
        nid = f"CN:{lbl}"
        G.add_node(nid, label=lbl, rel="related")
        G.add_edge(qid, nid)

    return G

# ——————————————————————————————————————————————————————————————————————————————
# 5. STREAMLIT UI
# ——————————————————————————————————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("🔗 Relationship-based Clustering")
st.markdown("""
1. Enter your **seed** topic  
2. Pull **Subtopics** (Wikidata P279) and **Related Entities** (ConceptNet)  
3. See three distinct bubbles: **Seed**, **Subtopics**, **Related Entities**  
""")

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Subtopic depth", 1, 2, 1)
tax_lim = st.slider("Max subtopics",   5, 50, 20)
sem_lim = st.slider("Max related items",5, 50, 20)

if st.button("Generate Graph"):
    G   = build_graph(seed, depth, tax_lim, sem_lim)
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # Build edge trace
    ex, ey = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(color="#888",width=1), hoverinfo="none")

    # Build node traces per relationship type
    traces = []
    for rel_type, color in [("seed","#ff6961"),("subtopic","#61ff8e"),("related","#61b2ff")]:
        xs, ys, txt = [], [], []
        for n,d in G.nodes(data=True):
            if d["rel"] == rel_type:
                x,y = pos[n]
                xs.append(x); ys.append(y); txt.append(d["label"])
        traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=txt,
                textposition="top center",
                marker=dict(size=20, color=color),
                name=rel_type.capitalize().replace("_"," "),
                hoverinfo="text"
            )
        )

    # Assemble figure
    fig = go.Figure(data=[edge_trace] + traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20,r=20,t=40,b=20),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)
    )

    st.plotly_chart(fig, use_container_width=True)
