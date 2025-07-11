import streamlit as st
import requests
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

# Load environment variables (API key and CSE ID)
load_dotenv()

# Get your API Key and Custom Search Engine ID from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
CSE_ID = st.secrets["CSE_ID"]

# Function to clean HTML from the snippet using BeautifulSoup
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

# Fetch Google Knowledge Graph data
def get_knowledge_graph_data(query):
    url = f'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'key': GOOGLE_API_KEY,
        'limit': 5,  # Number of results to return
        'indent': True
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'itemListElement' in data:
        return data['itemListElement']
    return []

# Fetch Wikidata data with relationships by specific Q-codes
def get_wikidata_entity_details(entity_id):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json"
    response = requests.get(url)
    data = response.json()
    
    if 'entities' in data:
        entity = data['entities'][entity_id]
        return entity
    return None

# Function to visualize knowledge graph using Plotly
def visualize_graph(google_entities, wikidata_entities):
    nodes = []
    edges = []
    
    # Add Google entities as nodes
    for entity in google_entities:
        nodes.append(entity['result']['name'])
    
    # Add Wikidata entities as nodes and relationships as edges
    for entity in wikidata_entities:
        nodes.append(entity['label'])
        for relationship in entity['relationships']:
            related_entity = relationship[1]
            related_entity_label = next(
                (e['label'] for e in wikidata_entities if e['id'] == related_entity), None)
            if related_entity_label:
                edges.append((entity['label'], related_entity_label))
    
    # Create node positions using a spring layout
    pos = {node: (i, j) for i, (node, j) in enumerate(zip(nodes, range(len(nodes))))}
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    
    # Create Plotly graph
    trace_nodes = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=nodes,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            color='skyblue',
        ),
        textposition='bottom center'
    )

    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge[0]], pos[edge[1]]
        edge_x.append(x0)
        edge_y.append(y0)
    
    trace_edges = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='none'
    )

    fig = go.Figure(data=[trace_edges, trace_nodes])

    fig.update_layout(
        title='Comprehensive Knowledge Graph Visualization',
        showlegend=False,
        hovermode='closest',
        template='plotly_dark',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    # Show the Plotly graph in Streamlit
    st.plotly_chart(fig)

# Streamlit interface
st.title("Comprehensive Knowledge Graph Visualization Tool")

# Input for entity or keyword
keyword = st.text_input("Enter an entity or keyword:")

if keyword:
    st.write(f"Fetching related entities for: {keyword}")
    
    # Fetch related entities from Google Knowledge Graph
    google_entities = get_knowledge_graph_data(keyword)
    st.write(f"Found {len(google_entities)} related entities from Google Knowledge Graph:")
    if not google_entities:
        st.write("No data returned from Google Knowledge Graph.")
    
    # Fetch Wikidata data for "Data lake" and "Data warehouse" by their Q-codes
    data_lake_entity = get_wikidata_entity_details('Q1234567')  # Replace with actual Q-code for "Data lake"
    data_warehouse_entity = get_wikidata_entity_details('Q9876543')  # Replace with actual Q-code for "Data warehouse"
    
    wikidata_entities = []
    if data_lake_entity:
        wikidata_entities.append(data_lake_entity)
    if data_warehouse_entity:
        wikidata_entities.append(data_warehouse_entity)
    
    st.write(f"Found {len(wikidata_entities)} related entities from Wikidata:")
    for entity in wikidata_entities:
        st.write(f"- Name: {entity['labels']['en']}, Description: {entity.get('descriptions', {}).get('en', 'N/A')}")
    
    # Visualize the knowledge graph if data is found
    if google_entities or wikidata_entities:
        visualize_graph(google_entities, wikidata_entities)
    else:
        st.write("No related entities found.")
