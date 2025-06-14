import streamlit as st

# Default key for local development
LOCAL_API_KEY = "e836f24d41df4ffbb53feda05259f35d"

# Try to get the key from Streamlit's secrets management
try:
    # This will work when deployed on Streamlit Cloud
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except (KeyError, FileNotFoundError):
    # This will be used for local development or if the secret is not set
    NEWS_API_KEY = LOCAL_API_KEY