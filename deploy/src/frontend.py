import streamlit as st
import pandas as pd
from requests import post
import os
from dotenv import load_dotenv

load_dotenv()

# Streamlit interface
def generate_interface():
    st.set_page_config(page_title="SQL AI", page_icon="✨", layout="centered")

    # Title and description
    st.title("SQL AI")
    st.markdown(
    """
    <style>
        div[data-testid="stColumn"]  {
                align-self: end;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # Create a two-column layout for input and button
    col1, col2 = st.columns([3, 1])  # 3 parts for input, 1 part for button
    with col1:
        query = st.text_input(label='Question',placeholder="Describe what you want to query...", value="List the names and capital of all companies with a company size described as MICRO EMPRESA.")
    with col2:
        generate_button = st.button("Generate ✨", type="primary")

    # Process the query when button is pressed
    if generate_button:
        with st.spinner("Generating SQL..."):
            api_url = os.getenv("API_URL")
            data = post(f"{api_url}/text-to-sql", json={"question": query}).json()
        
        
        # Display generated SQL query
        st.subheader("Generated SQL")
        st.code(data["sql_query"], language="sql")

        # Move results to the right sidebar by using st.sidebar
        with st.sidebar:
            st.subheader("Results")
            results = data["sql_response"]
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)

if __name__ == "__main__":
    generate_interface()
