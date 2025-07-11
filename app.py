import streamlit as st
import requests, re
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── CONFIG ────────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def lookup_qid(label: str) -> str | None:
    try:
        r = requests.get(WIKIDATA_API, params={
            "action":"wbsearchentities",
            "format":"json",
            "language":"en",
            "search": label,
            "limit": 1
        }, timeout=5)
        r.raise_for_status()
        hits = r.json().get("search", [])
        return hits[0]["id"] if hits else None
    except:
        return None

@st.cache_data
def get_subclasses(qid: str, limit: int=20):
    sparql = f"""
      SELECT ?child ?childLabel WHERE {{
        ?child wdt:P279 wd:{qid} .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }} LIMIT {limit}
    """
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": sparql},
        headers={"Accept":"application/sparql-results+json"},
        timeout=10
    ); r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    return [
        (row["child"]["value"].rsplit("/",1)[-1],
         row["childLabel"]["value"])
        for row in rows
    ]

@st.cache_data
def get_conceptnet_neighbors(term: str, limit: int=20):
    uri = term.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit},
        timeout=5
    ); r.raise_for_status()
    return [
        e["@id"].split("/")[-1].replace("_"," ")
        for e in r.json().get("related", [])
    ]

@st.cache_data
def get_gpt_neighbors(term: str, limit: int=10):
    prompt = (
        f"List {limit} concise, distinct search queries related to “{term}”. "
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

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_graph(seed, sub_depth, tax_lim, sem_lim, gpt_seed_lim, gpt_rel_lim):
    G = nx.Graph()
    # Seed
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # P279 subtopics
    qid = lookup_qid(seed)
    if qid:
        lvl1 = get_subclasses(qid, tax_lim)
        for cid, clbl in lvl1:
            G.add_node(clbl, label=clbl, rel="subtopic", depth=1)
            G.add_edge(seed, clbl)
        if sub_depth > 1:
            for cid, clbl in lvl1:
                cqid = lookup_qid(clbl)
                if not cqid: continue
                for c2id, c2lbl in get_subclasses(cqid, max(1, tax_lim//2)):
                    if not G.has_node(c2lbl):
                        G.add_node(c2lbl, label=c2lbl, rel="subtopic", depth=2)
                    G.add_edge(clbl, c2lbl)

    # ConceptNet related
    sems = get_conceptnet_neighbors(seed, sem_lim)
    for lbl in sems:
        G.add_node(lbl, label=lbl, rel="related", depth=1)
        G.add_edge(seed, lbl)

    # GPT on seed
    gpts = get_gpt_neighbors(seed, gpt_seed_lim)
    for qry in gpts:
        G.add_node(qry, label=qry, rel="gpt_seed", depth=1)
        G.add_edge(seed, qry)

    # GPT on each related
    for lbl in sems:
        subqs = get_gpt_neighbors(lbl, gpt_rel_lim)
        for sq in subqs:
            G.add_node(sq, label=sq, rel="gpt_related", depth=2)
            G.add_edge(lbl, sq)

    return G

# ─── VISUALIZE ───────────────────────────────────────────
def draw_pyvis(G: nx.Graph):
    net = Network(height="750px", width="100%", notebook=False)
    color_map = {
        "seed":        "#1f78b4",
        "subtopic":    "#66c2a5",
        "related":     "#61b2ff",
        "gpt_seed":    "#ffcc61",
        "gpt_related": "#ff61cc",
    }
    for nid, data in G.nodes(data=True):
        net.add_node(
            nid,
            label=data["label"],
            title=f"{data['rel']} (depth {data['depth']})",
            color=color_map[data["rel"]]
        )
    for u,v in G.edges():
        net.add_edge(u, v)
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── STREAMLIT UI ────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔗 Hybrid Wikidata/ConceptNet + GPT Clusters")

seed     = st.text_input("Seed topic", "data warehouse")
sub_d    = st.slider("Subtopic depth (P279)",      1, 2, 1)
tax_lim  = st.slider("Max subtopics",             5, 50, 20)
sem_lim  = st.slider("Max ConceptNet related",    5, 50, 20)
gpt_seed = st.slider("GPT queries on seed",       5, 50, 20)
gpt_rel  = st.slider("GPT queries on related",    2, 20, 5)

if st.button("Generate Graph"):
    with st.spinner("Building graph…"):
        G = build_graph(seed, sub_d, tax_lim, sem_lim, gpt_seed, gpt_rel)
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")
    # Legend
    st.markdown(
        "<span style='color:#1f78b4;'>🔵</span>Seed  "
        "<span style='color:#66c2a5;'>🟢</span>Subtopic  "
        "<span style='color:#61b2ff;'>🔷</span>Related  "
        "<span style='color:#ffcc61;'>🟠</span>GPT(seed)  "
        "<span style='color:#ff61cc;'>🟣</span>GPT(related)",
        unsafe_allow_html=True
    )
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
