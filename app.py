import streamlit as st
import requests
import os
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables (API key and CSE ID)
load_dotenv()

# Get your API Key and Custom Search Engine ID from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
CSE_ID = st.secrets["CSE_ID"]

# Function to clean HTML from the snippet using BeautifulSoup
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

# Function to fetch related entities from Google Custom Search API
def get_related_entities_from_google(keyword, num_results=100):
    related_entities = []
    for start_index in range(1, num_results + 1, 10):  # Paginate through results (10 per page)
        url = f'https://www.googleapis.com/customsearch/v1?q={keyword}&key={GOOGLE_API_KEY}&cx={CSE_ID}&start={start_index}'
        response = requests.get(url)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch data from Google. HTTP Status code: {response.status_code}")
            return []
        
        search_results = response.json().get('items', [])
        
        for result in search_results:
            cleaned_snippet = clean_html(result['snippet'])
            related_entities.append({
                'title': result['title'],
                'link': result['link'],
                'snippet': cleaned_snippet,
            })
    
    return related_entities

# Function to fetch related entities from Wikidata API
def get_related_entities_from_wikidata(keyword, num_results=50):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={keyword}&limit={num_results}&format=json"
    response = requests.get(url)
    entities = response.json().get('search', [])
    
    related_entities = []
    for entity in entities:
        related_entities.append({
            'id': entity['id'],
            'label': entity['label'],
            'description': entity.get('description', 'No description available')
        })
    return related_entities

# Function to visualize knowledge graph using Plotly
def visualize_graph(google_entities, wikidata_entities):
    # Initialize lists to hold nodes and edges
    nodes = []
    edges = []
    
    # Add Google entities as nodes
    for entity in google_entities:
        nodes.append(entity['title'])
        # Add edges to Google entities (if needed, based on relations)
        for wikidata_entity in wikidata_entities:
            if entity['title'].lower() == wikidata_entity['label'].lower():
                edges.append((entity['title'], wikidata_entity['label']))
    
    # Add Wikidata entities as nodes
    for entity in wikidata_entities:
        if entity['label'] not in nodes:
            nodes.append(entity['label'])
    
    # Create a mapping for nodes
    node_map = {node: i for i, node in enumerate(nodes)}
    
    # Create edge list for Plotly visualization
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = node_map[edge[0]], node_map[edge[1]]
        edge_x.append(x0)
        edge_y.append(y0)
    
    # Create node positions using a spring layout (force-directed graph)
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

    trace_edges = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='none'
    )

    fig = go.Figure(data=[trace_edges, trace_nodes])

    fig.update_layout(
        title='Knowledge Graph Visualization',
        showlegend=False,
        hovermode='closest',
        template='plotly_dark',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    # Show the Plotly graph in Streamlit
    st.plotly_chart(fig)

# Streamlit interface
st.title("Google Custom Search and Wikidata Knowledge Graph Visualization Tool")

# Input for entity or keyword
keyword = st.text_input("Enter an entity or keyword:")

if keyword:
    st.write(f"Fetching related entities for: {keyword}")
    
    # Fetch related entities from Google Custom Search
    google_entities = get_related_entities_from_google(keyword)
    
    # Fetch related entities from Wikidata
    wikidata_entities = get_related_entities_from_wikidata(keyword)
    
    # Check if entities were found
    if google_entities or wikidata_entities:
        if google_entities:
            st.write(f"Found {len(google_entities)} related entities from Google Custom Search:")
            for entity in google_entities:
                st.write(f"- {entity['title']}: {entity['snippet']}")
                st.write(f"  Link: {entity['link']}")
        
        if wikidata_entities:
            st.write(f"Found {len(wikidata_entities)} related entities from Wikidata:")
            for entity in wikidata_entities:
                st.write(f"- {entity['label']}: {entity.get('description', 'No description available')}")
        
        # Visualize the knowledge graph
        visualize_graph(google_entities, wikidata_entities)
    else:
        st.write("No related entities found.")
