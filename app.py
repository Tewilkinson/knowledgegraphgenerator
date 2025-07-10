import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import requests

# Function to get related entities from Wikipedia (instead of Wikidata)
def get_related_entities_from_wikipedia(keyword):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={keyword}&format=json"
    response = requests.get(url)
    search_results = response.json().get('query', {}).get('search', [])
    
    related_entities = []
    for result in search_results:
        related_entities.append({
            'title': result['title'],
            'snippet': result['snippet'],
        })
    return related_entities

# Function to visualize knowledge graph
def visualize_graph(entities):
    G = nx.Graph()
    
    # Add nodes and edges based on entities
    for entity in entities:
        G.add_node(entity['title'])
        # You can add connections (edges) based on additional logic or metadata
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
    
    # Show the plot in Streamlit
    st.pyplot(plt)

# Streamlit interface
st.title("Wikipedia Knowledge Graph Visualization Tool")
keyword = st.text_input("Enter an entity or keyword:")

if keyword:
    st.write(f"Fetching related entities from Wikipedia for: {keyword}")
    entities = get_related_entities_from_wikipedia(keyword)
    
    if entities:
        st.write(f"Found {len(entities)} related entities:")
        for entity in entities:
            st.write(f"- {entity['title']}: {entity['snippet']}")  # Show the title and snippet
            
        # Visualize the knowledge graph
        visualize_graph(entities)
    else:
        st.write("No related entities found.")
