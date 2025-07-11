import streamlit as st
import requests
import networkx as nx
from community import community_louvain   # pip install python-louvain
import plotly.graph_objects as go

# 1. LOOKUP: label → Wikidata QID
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label
        },
    )
    resp.raise_for_status()
    return resp.json()["search"][0]["id"]

# 2. TAXONOMY: direct subclasses (“fan-out”) via P279
@st.cache_data(show_spinner=False)
def get_subclasses(qid: str, limit: int = 20):
    sparql = f"""
    SELECT ?child ?childLabel WHERE {{
      ?child wdt:P279 wd:{qid} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    resp = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": sparql},
        headers={"Accept": "application/sparql-results+json"}
    )
    resp.raise_for_status()
    bindings = resp.json()["results"]["bindings"]
    return [
        (b["child"]["value"].rsplit("/",1)[-1], b["childLabel"]["value"])
        for b in bindings
    ]

# 3. SEMANTIC: related neighbours from ConceptNet
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    resp = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}
    )
    resp.raise_for_status()
    rels = resp.json().get("related", [])
    return [
        (e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0))
        for e in rels
    ]

# 4. BUILD the hybrid graph
def build_graph(seed: str, depth: int, tax_limit: int, sem_limit: int) -> nx.Graph:
    qid = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(qid, label=seed, type="seed")

    # 1-hop taxonomy
    for cq, cl in get_subclasses(qid, tax_limit):
        G.add_node(cq, label=cl, type="taxonomy")
        G.add_edge(qid, cq, type="taxonomy")

    # 1-hop semantic
    for lbl, _ in get_conceptnet_neighbors(seed, sem_limit):
        nid = f"CN:{lbl}"
        G.add_node(nid, label=lbl, type="semantic")
        G.add_edge(qid, nid, type="semantic")

    # optional 2-hop taxonomy
    if depth > 1:
        for n,d in list(G.nodes(data=True)):
            if d["type"] == "taxonomy":
                for cq, cl in get_subclasses(n, tax_limit//2):
                    if not G.has_node(cq):
                        G.add_node(cq, label=cl, type="taxonomy")
                    G.add_edge(n, cq, type="taxonomy")

    return G

# 5. STREAMLIT UI
st.set_page_config(layout="wide")
st.title("🔗 Hybrid Knowledge-Graph Content Clusters")
st.markdown(
    """
    Enter a **seed topic**, fetch both  
    • **fan-out** (Wikidata subclasses)  
    • **semantic** neighbours (ConceptNet)  
    …and visualize them in colored, grouped clusters.
    """
)

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Taxonomy depth (fan-out layers)", 1, 2, 1)
tax_lim = st.slider("Max subclasses (Wikidata P279)", 5, 50, 20)
sem_lim = st.slider("Max related neighbours (ConceptNet)", 5, 50, 20)

if st.button("Generate Graph"):
    with st.spinner("Building and clustering…"):
        G = build_graph(seed, depth, tax_lim, sem_lim)
        partition = community_louvain.best_partition(G)

    # 6. LAYOUT + PLOTLY TRACES
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Edge trace
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none"
    )

    # Node traces, one per cluster
    node_traces = []
    for cluster_id in set(partition.values()):
        xs, ys, labels = [], [], []
        for n, d in G.nodes(data=True):
            if partition[n] == cluster_id:
                x, y = pos[n]
                xs.append(x); ys.append(y); labels.append(d["label"])
        node_traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=20),
                name=f"Cluster {cluster_id}",
                hoverinfo="text"
            )
        )

    # Assemble figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)
