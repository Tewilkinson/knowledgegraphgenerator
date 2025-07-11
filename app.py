import streamlit as st
import requests, re
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── CONFIG ────────────────────────────────────────────
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def lookup_qid(label: str) -> str | None:
    """Wikidata search API → QID."""
    try:
        r = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "search": label,
                "limit": 1
            },
            timeout=5
        )
        r.raise_for_status()
        hits = r.json().get("search", [])
        return hits[0]["id"] if hits else None
    except:
        return None

@st.cache_data(show_spinner=False)
def get_subclasses(qid: str, limit: int = 20) -> list[tuple[str,str]]:
    """Wikidata P279 subclasses → list of (QID, label)."""
    sparql = f"""
      SELECT ?child ?childLabel WHERE {{
        ?child wdt:P279 wd:{qid} .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }} LIMIT {limit}
    """
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": sparql},
        headers={"Accept": "application/sparql-results+json"},
        timeout=10
    ); r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [
        (row["child"]["value"].rsplit("/",1)[-1],
         row["childLabel"]["value"])
        for row in rows
    ]

@st.cache_data(show_spinner=False)
def get_conceptnet_neighbors(concept: str, limit: int = 20) -> list[str]:
    """ConceptNet related neighbours → list of labels."""
    uri = concept.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit},
        timeout=5
    ); r.raise_for_status()
    return [
        e["@id"].split("/")[-1].replace("_"," ")
        for e in r.json().get("related", [])
    ]

@st.cache_data(show_spinner=False)
def get_gpt_neighbors(seed: str, limit: int = 20) -> list[str]:
    """GPT → list of related search queries."""
    prompt = (
        f"List {limit} concise, distinct search queries related to “{seed}”. "
        "Return them as a bulleted list, one per line."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7
    )
    out = []
    for ln in resp.choices[0].message.content.splitlines():
        clean = re.sub(r"^[-•\s]+","", ln.strip())
        if clean: out.append(clean)
    return out

# ─── BUILD GRAPH ────────────────────────────────────────
def build_graph(seed, depth, tax_lim, sem_lim, gpt_lim):
    G = nx.Graph()
    # Seed node
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # 1-hop Wikidata subtopics
    qid = lookup_qid(seed)
    if qid:
        for cid, clbl in get_subclasses(qid, tax_lim):
            G.add_node(clbl, label=clbl, rel="subtopic", depth=1)
            G.add_edge(seed, clbl)

    # 2-hop Wikidata subtopics
    if depth > 1:
        first = [n for n,d in G.nodes(data=True) if d["rel"]=="subtopic"]
        for parent in first:
            parent_qid = lookup_qid(parent)
            if not parent_qid: continue
            for cid, clbl in get_subclasses(parent_qid, max(1,tax_lim//2)):
                if not G.has_node(clbl):
                    G.add_node(clbl, label=clbl, rel="subtopic", depth=2)
                G.add_edge(parent, clbl)

    # 1-hop ConceptNet neighbours
    for lbl in get_conceptnet_neighbors(seed, sem_lim):
        G.add_node(lbl, label=lbl, rel="related", depth=1)
        G.add_edge(seed, lbl)

    # 1-hop GPT queries
    for qry in get_gpt_neighbors(seed, gpt_lim):
        G.add_node(qry, label=qry, rel="gpt", depth=1)
        G.add_edge(seed, qry)

    return G

# ─── RENDER WITH PYVIS ───────────────────────────────────
def draw_pyvis(G: nx.Graph):
    net = Network(height="750px", width="100%", notebook=False)
    color_map = {
        "seed":    "#1f78b4",
        "subtopic":"#66c2a5",
        "related": "#61b2ff",
        "gpt":     "#ffcc61",
    }
    for nid, data in G.nodes(data=True):
        net.add_node(
            nid,
            label=data["label"],
            title=f"{data['rel']} (depth {data['depth']})",
            color=color_map.get(data["rel"], "#ccc")
        )
    for u,v in G.edges():
        net.add_edge(u, v)
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 Hybrid Wikidata/ConceptNet/GPT Graph")

seed    = st.text_input("Seed topic", "data warehouse")
depth   = st.slider("Subtopic depth (Wikidata)", 1, 2, 1)
tax_lim = st.slider("Max subtopics (P279)",      5, 50, 20)
sem_lim = st.slider("Max ConceptNet items",      5, 50, 20)
gpt_lim = st.slider("Max GPT queries",           5, 50, 20)

if st.button("Generate Graph"):
    with st.spinner("Building graph…"):
        G = build_graph(seed, depth, tax_lim, sem_lim, gpt_lim)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    legend = "<span style='color:#1f78b4;'>🔵</span>Seed  " + \
             "<span style='color:#66c2a5;'>🟢</span>Subtopic  " + \
             "<span style='color:#61b2ff;'>🔷</span>Related  " + \
             "<span style='color:#ffcc61;'>🟠</span>GPT"
    st.markdown(legend, unsafe_allow_html=True)
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
