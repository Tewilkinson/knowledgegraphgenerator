import streamlit as st
import requests

# Replace with your Google API Key
API_KEY = st.secrets["GOOGLE_API_KEY"]

def get_knowledge_graph_data(query):
    url = f'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'key': API_KEY,
        'limit': 5,
        'indent': True
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'itemListElement' in data:
        return data['itemListElement']
    return []

# Streamlit UI
st.title("Google Knowledge Graph Visualization Tool")

# Input for entity or keyword
keyword = st.text_input("Enter an entity or keyword:")

if keyword:
    st.write(f"Fetching related entities for: {keyword}")
    
    # Fetch related entities from Google's Knowledge Graph API
    entities = get_knowledge_graph_data(keyword)
    
    if entities:
        st.write(f"Found {len(entities)} related entities from Google Knowledge Graph:")
        for entity in entities:
            result = entity['result']
            st.write(f"- Name: {result.get('name', 'N/A')}")
            st.write(f"  Description: {result.get('description', 'N/A')}")
            st.write(f"  Types: {result.get('type', 'N/A')}")
            st.write(f"  URL: {result.get('url', 'N/A')}")
            st.write("-" * 50)
    else:
        st.write("No related entities found.")
