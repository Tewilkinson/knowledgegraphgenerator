import streamlit as st
import requests

# Replace with your Google API Key
API_KEY = st.secrets["GOOGLE_API_KEY"]

def get_knowledge_graph_data(query):
    url = f'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'key': API_KEY,
        'limit': 5,  # Number of results to return
        'indent': True
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Print the entire response to inspect the data structure
    st.write(data)
    
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
            
            # Check if data is available for each field, else display 'Data not available'
            name = result.get('name', 'Data not available')
            description = result.get('description', 'Description not available')
            types = result.get('type', 'Types not available')
            url = result.get('url', 'URL not available')
            
            st.write(f"- Name: {name}")
            st.write(f"  Description: {description}")
            st.write(f"  Types: {types}")
            st.write(f"  URL: {url}")
            st.write("-" * 50)
    else:
        st.write("No related entities found.")
