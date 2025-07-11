# app.py

import streamlit as st
import requests, re, csv, os, io, zipfile
import pandas as pd
import networkx as nx
from pyvis.network import Network
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI

# ─── CONFIG ────────────────────────────────────────────
openai_client   = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

ONTOLOGY = {
    "hierarchy_predicates":   ["P279", "P31"],        # candidate subtopics
    "association_predicates": ["P361", "P527", "P921"] # always related
}

COLOR_MAP = {
    "seed":     "#1f78b4",
    "subtopic": "#33a02c",
    "related":  "#ff7f00",
    "gpt":      "#e31a1c",
}

# ─── UI ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔎 Simple Hybrid Knowledge Graph")

with st.sidebar:
    seed     = st.text_input("Seed entity", "data warehouse")
    depth    = st.slider("Wikidata recurse depth", 1, 2, 1)
    tax_lim  = st.slider("Max P279/P31 items", 5, 50, 20)
    sem_lim  = st.slider("Max ConceptNet related", 5, 50, 20)
    gpt_seed = st.slider("GPT on seed", 5, 50, 20)
    gpt_rel  = st.slider("GPT on related", 2, 20, 5)
    build    = st.button("Build Graph")

# legend
st.markdown(
    "<span style='color:" + COLOR_MAP["seed"] + "'>●</span> Seed  "
    "<span style='color:" + COLOR_MAP["subtopic"] + "'>●</span> Subtopic  "
    "<span style='color:" + COLOR_MAP["related"] + "'>●</span> Related  "
    "<span style='color:" + COLOR_MAP["gpt"] + "'>●</span> GPT",
    unsafe_allow_html=True
)

# ─── HELPERS ────────────────────────────────────────────
@st.cache_data
def lookup_qid(label: str) -> str | None:
    try:
        r = requests.get(
            WIKIDATA_API,
            params={"action":"wbsearchentities","format":"json",
                    "language":"en","search":label,"limit":1},
            timeout=5
        )
        r.raise_for_status()
        hits = r.json().get("search", [])
        return hits[0]["id"] if hits else None
    except:
        return None

@st.cache_data
def get_wikidata_relations(qid: str, preds: list[str]):
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    vals   = " ".join(f"wdt:{p}" for p in preds)
    sparql.setQuery(f"""
      SELECT ?p ?objLabel WHERE {{
        VALUES ?p {{ {vals} }}
        wd:{qid} ?p ?obj .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      }}
    """)
    sparql.setReturnFormat(JSON)
    rows = sparql.query().convert()["results"]["bindings"]
    return [
        (r["p"]["value"].split("/")[-1], r["objLabel"]["value"])
        for r in rows
    ]

@st.cache_data
def get_conceptnet_related(term: str, limit: int=20):
    uri = term.lower().replace(" ", "_")
    r = requests.get(
        f"https://api.conceptnet.io/related/c/en/{uri}",
        params={"filter":"/c/en","limit":limit}, timeout=5
    )
    r.raise_for_status()
    return [e["@id"].split("/")[-1].replace("_"," ")
            for e in r.json().get("related", [])]

@st.cache_data
def get_gpt_related(term: str, limit: int=10):
    prompt = (
        f"List {limit} concise, distinct search queries related to “{term}”. "
        "Return as a bulleted list, one per line."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.7
    )
    return [
        re.sub(r"^[-•\s]+","", ln.strip())
        for ln in resp.choices[0].message.content.splitlines()
        if ln.strip()
    ]

@st.cache_data
def classify_edge_with_gpt(parent: str, child: str) -> str:
    prompt = (
        f"Given concept **{parent}** and candidate subtopic **{child}**, "
        "is this a subtopic or just related? Answer: subtopic or related."
    )
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are a precise classifier."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0
    )
    ans = resp.choices[0].message.content.strip().lower()
    return "subtopic" if "subtopic" in ans else "related"

# ─── BUILD GRAPH ─────────────────────────────────────────
def build_graph():
    G = nx.Graph()
    # seed
    G.add_node(seed, label=seed, rel="seed", depth=0)

    # Wikidata
    qid = lookup_qid(seed)
    if qid:
        def recurse(label, qid, d):
            if d > depth: return
            for prop, obj in get_wikidata_relations(qid, ONTOLOGY["hierarchy_predicates"] + ONTOLOGY["association_predicates"]):
                # classify
                if prop in ONTOLOGY["hierarchy_predicates"]:
                    rel = classify_edge_with_gpt(label, obj)
                else:
                    rel = "related"
                G.add_node(obj, label=obj, rel=rel, depth=d)
                G.add_edge(label, obj)
                if rel == "subtopic":
                    # only recurse true subtopics
                    recurse(obj, lookup_qid(obj) or "", d+1)
        recurse(seed, qid, 1)

    # ConceptNet
    for related in get_conceptnet_related(seed, sem_lim):
        G.add_node(related, label=related, rel="related", depth=1)
        G.add_edge(seed, related)

    # GPT on seed
    for qry in get_gpt_related(seed, gpt_seed):
        G.add_node(qry, label=qry, rel="gpt", depth=1)
        G.add_edge(seed, qry)

    # GPT on each related/subtopic
    for node, data in list(G.nodes(data=True)):
        if data["rel"] in ("related","subtopic"):
            for sub in get_gpt_related(node, gpt_rel):
                G.add_node(sub, label=sub, rel="gpt", depth=data["depth"]+1)
                G.add_edge(node, sub)

    return G

# ─── RENDER & EXPORT ─────────────────────────────────────
def draw_pyvis(G):
    net = Network(height="750px", width="100%", notebook=False)
    for n,d in G.nodes(data=True):
        net.add_node(n, label=d["label"],
                     title=f"{d['rel']} depth={d['depth']}",
                     color=COLOR_MAP[d["rel"]])
    for u,v in G.edges():
        net.add_edge(u,v)
    net.show_buttons(filter_=['physics'])
    return net.generate_html()

if build:
    with st.spinner("Building graph…"):
        G = build_graph()
    st.success(f"Nodes: {len(G.nodes)}  Edges: {len(G.edges)}")

    # export
    df_nodes = pd.DataFrame([
        {"node":n, "label":d["label"], "rel":d["rel"], "depth":d["depth"]}
        for n,d in G.nodes(data=True)
    ])
    df_edges = pd.DataFrame([
        {"source":u, "target":v} for u,v in G.edges()
    ])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nodes.csv", df_nodes.to_csv(index=False))
        z.writestr("edges.csv", df_edges.to_csv(index=False))
    buf.seek(0)

    st.download_button("Download ZIP", buf, "graph.zip", "application/zip")
    st.components.v1.html(draw_pyvis(G), height=750, scrolling=True)
