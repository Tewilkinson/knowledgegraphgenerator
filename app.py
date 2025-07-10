import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import requests

# Function to get related entities from Wikidata (as an example)
def get_related_entities(keyword):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={keyword}&limit=10&format=json"
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
def visualize_graph(entities):
    G = nx.Graph()
    
    # Add nodes and edges based on entities and their connections
    for entity in entities:
        G.add_node(entity['label'])
        # In a real case, you'd add connections (edges) here based on the relationships
        
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
    
    # Show the plot in Streamlit
    st.pyplot(plt)

# Streamlit interface
st.title("Knowledge Graph Visualization Tool")
keyword = st.text_input("Enter an entity or keyword:")

if keyword:
    st.write(f"Fetching entities related to: {keyword}")
    entities = get_related_entities(keyword)
    
    if entities:
        st.write(f"Found {len(entities)} related entities:")
        for entity in entities:
            st.write(f"- {entity['label']}: {entity['description']}")
        
        # Visualize the knowledge graph
        visualize_graph(entities)
    else:
        st.write("No related entities found.")
