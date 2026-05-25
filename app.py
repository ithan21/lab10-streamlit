import streamlit as st
import requests
from PIL import Image
import io

# Kuhain ang token mula sa secrets.toml
HF_TOKEN = "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"

# --- FUNCTIONS ---
def get_chat_response(prompt):
    """Magpadala ng message sa Chat AI"""
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        elif response.status_code == 503:
            return "⏳ The AI model is waking up. Please try again in 20 seconds."
        else:
            return "❌ Error communicating with the AI."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_image(prompt):
    """Mag-generate ng image gamit API"""
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        elif response.status_code == 503:
            st.error("⏳ The Image model is loading in the server. Please wait 20 seconds and click Generate again.")
            return None
        else:
            st.error("❌ Failed to generate image.")
            return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# --- UI DESIGN ---
st.set_page_config(page_title="Lab 10 AI", page_icon="🤖")

st.title("🤖 Multi-Modal AI Application")
st.caption("Lab 10.0 - Natural Language Processing (Large Language Models)")
st.divider()

tab1, tab2 = st.tabs(["💬 Chat", "🖼️ Image Generator"])

# --- TAB 1: CHAT ---
with tab1:
    # I-save ang chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # I-display ang mga nakaraang messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input box para sa user
    if user_input := st.chat_input("Type your message here..."):
        # Idisplay ang mensahe ng user
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Kunin ang sagot ng AI
        with st.spinner("Thinking..."):
            bot_reply = get_chat_response(user_input)

        # Idisplay ang sagot ng AI
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

# --- TAB 2: IMAGE GENERATOR ---
with tab2:
    img_prompt = st.text_area("Describe the image you want to create:", height=100, placeholder="e.g., A cute cat astronaut in space, digital art")
    
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if img_prompt:
            with st.spinner("Generating image (might take 10-20 seconds on first try)..."):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

# Footer
st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
