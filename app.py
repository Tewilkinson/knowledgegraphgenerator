# app.py

import streamlit as st
import requests, re, csv, os, io
import pandas as pd
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── 1. CONFIG & ONTOLOGY SCHEMA ──────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

ONTOLOGY = {
    "hierarchy_predicates": ["P279", "P31"],
    "association_predicates": ["P361", "P527", "P921"]
}

# ─── 2. STREAMLIT UI ───────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Enhanced Hybrid Knowledge Graph")

with st.sidebar:
    seed       = st.text_input("Seed entity", "data warehouse")
    sub_depth  = st.slider("Subtopic depth (Wikidata)", 1, 2, 1)
    tax_lim    = st.slider("Max P279/P31 per node", 5, 50, 20)
    sem_lim    = st.slider("Max ConceptNet related", 5, 50, 20)
    gpt_seed   = st.slider("GPT queries on seed", 5, 50, 20)
    gpt_rel    = st.slider("GPT on each related node", 2, 20, 5)
    build      = st.button("Build Graph")

st.markdown("""
🔵 Seed  
🟢 Custom Hierarchy  
🟡 Wikidata Hierarchy  
🟣 Wikidata Association  
🔷 ConceptNet  
🟠 GPT (seed)  
🟣 GPT (related)
""")

# ─── 3. HELPERS ────────────────────────────────────────────────────────────
@st.cache_data
def lookup_qid(label: str) -> str | None:
    try:
        r = requests.get(WIKIDATA_API, params={
            "action":"wbsearchentities","format":"json",
            "language":"en","search":label,"limit":1
        }, timeout=5)
        r.raise_for_status()
        hits = r.json().get("search", [])
        return hits[0]["id"] if hits else None
    except:
        return None

@st.cache_data
def get_wikidata_relations(qid: str, preds: list[str]):
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    vals    = " ".join(f"wdt:{p}" for p in preds)
    sparql.setQuery(f"""
      SELECT ?p ?pLabel ?objLabel WHERE {{
        VALUES ?p {{ {vals} }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    rows = sparql.query().convert()["results"]["bindings"]
    return [
        (r["p"]["value"].split("/")[-1],
         r["pLabel"]["value"],
         r["objLabel"]["value"])
        for r in rows
    ]

@st.cache_data
def get_conceptnet_related(term: str, limit: int=20):
    uri = term.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}, timeout=5
    ); r.raise_for_status()
    return [ e["@id"].split("/")[-1].replace("_"," ")
             for e in r.json().get("related", []) ]

@st.cache_data
def get_gpt_related(term: str, limit: int=10):
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
        if clean:
            out.append(clean)
    return out

@st.cache_data
def classify_edge_with_gpt(parent: str, child: str) -> str:
    prompt = (
        f"Given a domain concept **{parent}** and a candidate subtopic **{child}**, "
        "should this be classified as a subtopic or merely related? "
        "Answer one word: subtopic or related."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a precise classifier."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
    )
    ans = resp.choices[0].message.content.strip().lower()
    return "hierarchy" if "subtopic" in ans else "association"

def load_custom_hierarchy(csv_path="custom_hierarchy.csv"):
    edges = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    edges.append((row[0].strip(), row[1].strip()))
    return edges

# ─── 4. BUILD GRAPH ───────────────────────────────────────────────────────
def build_graph():
    G = nx.Graph()
    custom = load_custom_hierarchy()

    # Seed
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # Custom overrides
    for parent, child in custom:
        if not G.has_node(parent):
            G.add_node(parent, label=parent, rel="seed", depth=0)
        G.add_node(child, label=child, rel="custom", depth=1)
        G.add_edge(parent, child)

    # Wikidata relations
    qid = lookup_qid(seed)
    if qid:
        wrels = get_wikidata_relations(
            qid,
            ONTOLOGY["hierarchy_predicates"]
            + ONTOLOGY["association_predicates"]
        )
        for prop, _, obj_lbl in wrels:
            if prop in ONTOLOGY["hierarchy_predicates"]:
                rtype = classify_edge_with_gpt(seed, obj_lbl)
            else:
                rtype = "association"
            G.add_node(obj_lbl, label=obj_lbl, rel=rtype, depth=1)
            G.add_edge(seed, obj_lbl)

    # ConceptNet neighbours
    sems = get_conceptnet_related(seed, sem_lim)
    for lbl in sems:
        G.add_node(lbl, label=lbl, rel="related", depth=1)
        G.add_edge(seed, lbl)

    # GPT on seed
    for qry in get_gpt_related(seed, gpt_seed):
        G.add_node(qry, label=qry, rel="gpt_seed", depth=1)
        G.add_edge(seed, qry)

    # GPT on each related/association/custom node
    for node in list(G.nodes()):
        data = G.nodes[node]
        if data.get("rel") in ("related", "association", "custom"):
            for sub in get_gpt_related(node, gpt_rel):
                G.add_node(sub, label=sub, rel="gpt_related",
                           depth=data.get("depth", 0) + 1)
                G.add_edge(node, sub)

    return G

# ─── 5. RENDER & EXPORT ─────────────────────────────────────────────────
def draw_pyvis(G):
    net = Network(height="750px", width="100%", notebook=False)
    color_map = {
        "seed":        "#1f78b4",
        "custom":      "#2ca02c",
        "hierarchy":   "#ff7f0e",
        "association": "#9467bd",
        "related":     "#1f77b4",
        "gpt_seed":    "#d62728",
        "gpt_related": "#e377c2"
    }
    for nid, data in G.nodes(data=True):
        net.add_node(
            nid,
            label=data["label"],
            title=f"{data['rel']} (depth {data['depth']})",
            color=color_map.get(data.get("rel"), "#ccc")
        )
    for u, v in G.edges():
        net.add_edge(u, v)
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

# ─── 6. MAIN ─────────────────────────────────────────────────────────────
if build:
    with st.spinner("Building enhanced knowledge graph…"):
        G = build_graph()
    st.success(f"✅ Nodes: {len(G.nodes)}   Edges: {len(G.edges)}")

    # --- Export nodes to CSV ---
    df_nodes = pd.DataFrame([
        {"node_id": nid, "label": data.get("label",""),
         "rel": data.get("rel",""), "depth": data.get("depth",0)}
        for nid, data in G.nodes(data=True)
    ])
    csv_nodes = df_nodes.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download nodes CSV",
        data=csv_nodes,
        file_name="graph_nodes.csv",
        mime="text/csv"
    )

    # --- Export edges to CSV ---
    df_edges = pd.DataFrame([
        {"source": u, "target": v}
        for u, v in G.edges()
    ])
    csv_edges = df_edges.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download edges CSV",
        data=csv_edges,
        file_name="graph_edges.csv",
        mime="text/csv"
    )

    # Render the graph
    html = draw_pyvis(G)
    st.components.v1.html(html, height=800, scrolling=True)
