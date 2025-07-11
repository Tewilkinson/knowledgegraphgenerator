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

# Fetch Wikidata data with relationships
def get_related_entities_from_wikidata(keyword, num_results=50):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={keyword}&limit={num_results}&format=json"
    response = requests.get(url)
    entities = response.json().get('search', [])
    
    related_entities = []
    for entity in entities:
        entity_id = entity['id']
        details_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&format=json"
        details_response = requests.get(details_url)
        entity_details = details_response.json().get('entities', {}).get(entity_id, {})
        
        claims = entity_details.get('claims', {})
        relationships = []
        
        if 'P31' in claims:  # Instance of (e.g., Human)
            for claim in claims['P31']:
                related_entity_id = claim['mainsnak']['datavalue']['value']['id']
                relationships.append(('instance_of', related_entity_id))

        if 'P69' in claims:  # Educated at (e.g., Princeton University)
            for claim in claims['P69']:
                related_entity_id = claim['mainsnak']['datavalue']['value']['id']
                relationships.append(('educated_at', related_entity_id))
        
        related_entities.append({
            'id': entity_id,
            'label': entity['label'],
            'description': entity.get('description', 'No description available'),
            'relationships': relationships
        })
    
    return related_entities

# Fetch DBpedia data for related entities (semantic data)
def get_related_entities_from_dbpedia(keyword):
    url = f"http://dbpedia.org/sparql?query=SELECT%20%3Fsubject%20%3Fpredicate%20%3Fobject%20WHERE%20{{%20?subject%20?predicate%20?object%20FILTER%20(REGEX(str(?subject),%20'{keyword}'))%20}}&format=json"
    response = requests.get(url)
    data = response.json()
    
    related_entities = []
    for result in data.get('results', {}).get('bindings', []):
        subject = result.get('subject', {}).get('value')
        predicate = result.get('predicate', {}).get('value')
        object_ = result.get('object', {}).get('value')
        
        related_entities.append({
            'subject': subject,
            'predicate': predicate,
            'object': object_
        })
    
    return related_entities

# Visualize the knowledge graph using Plotly
def visualize_graph(google_entities, wikidata_entities, dbpedia_entities):
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
    
    # Add DBpedia relationships as edges
    for entity in dbpedia_entities:
        subject = entity['subject']
        object_ = entity['object']
        edges.append((subject, object_))
    
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
    
    # Fetch related entities from Wikidata
    wikidata_entities = get_related_entities_from_wikidata(keyword)
    
    # Fetch related entities from DBpedia
    dbpedia_entities = get_related_entities_from_dbpedia(keyword)
    
    if google_entities or wikidata_entities or dbpedia_entities:
        st.write(f"Found {len(google_entities)} related entities from Google Knowledge Graph:")
        for entity in google_entities:
            st.write(f"- Name: {entity['result']['name']}")
        
        st.write(f"Found {len(wikidata_entities)} related entities from Wikidata:")
        for entity in wikidata_entities:
            st.write(f"- Name: {entity['label']}, Description: {entity.get('description', 'N/A')}")
        
        st.write(f"Found {len(dbpedia_entities)} relationships from DBpedia:")
        for entity in dbpedia_entities:
            st.write(f"- {entity['subject']} {entity['predicate']} {entity['object']}")
        
        # Visualize the knowledge graph
        visualize_graph(google_entities, wikidata_entities, dbpedia_entities)
    else:
        st.write("No related entities found.")
