import streamlit as st
import requests

st.title("🔍 API Connection Test")
st.write("Nagte-test ng koneksyon sa Hugging Face...")

HF_TOKEN = "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"

st.subheader("1. Testing Chat Model...")
try:
    resp = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": "Say hello"},
        timeout=120
    )
    st.json(resp.json()) # Ito ang magpapakita ng exact error kung may mali
except Exception as e:
    st.error(f"Network Error: {repr(e)}")

st.divider()

st.subheader("2. Testing Image Model...")
try:
    resp = requests.post(
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": "a cat"},
        timeout=120
    )
    if resp.status_code == 200:
        st.success("✅ Image Model Works!")
    else:
        st.json(resp.json()) # Ito ang magpapakita ng exact error kung may mali
except Exception as e:
    st.error(f"Network Error: {repr(e)}")
