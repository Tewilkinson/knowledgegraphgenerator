import streamlit as st
import requests
import os
import networkx as nx
import matplotlib.pyplot as plt
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

# Function to visualize knowledge graph
def visualize_graph(google_entities, wikidata_entities):
    G = nx.Graph()
    
    # Add Google entities as nodes
    for entity in google_entities:
        G.add_node(entity['title'], type='google')
    
    # Add Wikidata entities as nodes
    for entity in wikidata_entities:
        G.add_node(entity['label'], type='wikidata')
    
    # Add edges (for now, we add edges between Google and Wikidata entities with the same name)
    for google_entity in google_entities:
        for wikidata_entity in wikidata_entities:
            if google_entity['title'].lower() == wikidata_entity['label'].lower():
                G.add_edge(google_entity['title'], wikidata_entity['label'])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
    
    # Show the plot in Streamlit
    st.pyplot(plt)

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
