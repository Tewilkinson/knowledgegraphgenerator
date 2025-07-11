import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go

# ——————————————————————————————————————————————————————————————————————————————
# 1. LOOKUP: label → Wikidata QID
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str:
    resp = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={"action":"wbsearchentities","format":"json","language":"en","search":label}
    )
    resp.raise_for_status()
    return resp.json()["search"][0]["id"]

# ——————————————————————————————————————————————————————————————————————————————
# 2. TAXONOMY: Fetch direct subclasses (fan-out) via P279
# ——————————————————————————————————————————————————————————————————————————————
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
        params={"query":sparql},
        headers={"Accept":"application/sparql-results+json"}
    )
    resp.raise_for_status()
    results = resp.json()["results"]["bindings"]
    return [
        (r["child"]["value"].rsplit("/",1)[-1], r["childLabel"]["value"])
        for r in results
    ]

# ——————————————————————————————————————————————————————————————————————————————
# 3. SEMANTIC: Fetch ConceptNet neighbours
# ——————————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20):
    uri = concept.lower().replace(" ", "_")
    resp = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}
    )
    resp.raise_for_status()
    rels = resp.json().get("related", [])
    return [(e["@id"].split("/")[-1].replace("_"," "), e.get("weight",0)) for e in rels]

# ——————————————————————————————————————————————————————————————————————————————
# 4. BUILD GRAPH
# ——————————————————————————————————————————————————————————————————————————————
def build_graph(seed: str, depth: int, tax_limit: int, sem_limit: int):
    qid = lookup_qid(seed)
    G = nx.Graph()
    G.add_node(qid, label=seed, type="seed")

    # 1-hop fan-out
    for child_q, child_lbl in get_subclasses(qid, tax_limit):
        G.add_node(child_q, label=child_lbl, type="taxonomy")
        G.add_edge(qid, child_q, type="taxonomy")

    # 1-hop related
    for lbl, _ in get_conceptnet_neighbors(seed, sem_limit):
        nid = f"CN:{lbl}"
        G.add_node(nid, label=lbl, type="semantic")
        G.add_edge(qid, nid, type="semantic")

    # optional 2-hop fan-out
    if depth > 1:
        for n, d in list(G.nodes(data=True)):
            if d["type"]=="taxonomy":
                for cq, cl in get_subclasses(n, tax_limit//2):
                    if not G.has_node(cq):
                        G.add_node(cq, label=cl, type="taxonomy")
                    G.add_edge(n, cq, type="taxonomy")
    return G

# ——————————————————————————————————————————————————————————————————————————————
# 5. STREAMLIT UI
# ——————————————————————————————————————————————————————————————————————————————
st.set_page_config(layout="wide")
st.title("🔗 Clustered Knowledge-Graph Explorer")
st.markdown(
    """
    1. Enter your **seed** topic  
    2. Pull **fan-out** (Wikidata subclasses) + **semantic** (ConceptNet) neighbours  
    3. See one bubble per first-level subclass + one bubble of **Related** topics  
    4. Zoom & pan interactively via Plotly
    """
)

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Fan-out depth", 1, 2, 1)
tax_lim = st.slider("Max fan-out nodes",  5, 50, 20)
sem_lim = st.slider("Max related nodes",   5, 50, 20)

if st.button("Generate Graph"):
    G = build_graph(seed, depth, tax_lim, sem_lim)
    seed_q = lookup_qid(seed)

    # ——————————————————————————————————————————————————————————————————
    # 6. CLUSTER ASSIGNMENT
    #    - seed → its own cluster
    #    - each first‐level taxonomy child → cluster named after itself
    #    - all semantic neighbours → "Related Topics" cluster
    # ——————————————————————————————————————————————————————————————————
    cluster_map = {}
    for n, d in G.nodes(data=True):
        if n == seed_q:
            cluster_map[n] = seed
        elif d["type"] == "taxonomy":
            # if direct child of seed, cluster = its own label
            if seed_q in G[n]:
                cluster_map[n] = G.nodes[n]["label"]
            else:
                # second‐level, find its first‐level ancestor
                path = nx.shortest_path(G, seed_q, n)
                first = path[1]
                cluster_map[n] = G.nodes[first]["label"]
        else:
            cluster_map[n] = "Related Topics"

    # ——————————————————————————————————————————————————————————————————
    # 7. LAYOUT via NetworkX spring
    # ——————————————————————————————————————————————————————————————————
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # ——————————————————————————————————————————————————————————————————
    # 8. Build the edge trace
    # ——————————————————————————————————————————————————————————————————
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#888", width=1),
        hoverinfo="none"
    )

    # ——————————————————————————————————————————————————————————————————
    # 9. Build one node‐trace per cluster
    # ——————————————————————————————————————————————————————————————————
    node_traces = []
    for cluster in sorted(set(cluster_map.values())):
        xs, ys, labels = [], [], []
        for n, d in G.nodes(data=True):
            if cluster_map[n] == cluster:
                x, y = pos[n]
                xs.append(x); ys.append(y); labels.append(d["label"])
        node_traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=18),
                name=cluster,
                hoverinfo="text"
            )
        )

    # ——————————————————————————————————————————————————————————————————
    # 10. Assemble & Display
    # ——————————————————————————————————————————————————————————————————
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
