import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import time

HF_TOKEN = "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"

client = InferenceClient(token=HF_TOKEN, timeout=120)

def get_chat_response(prompt, max_retries=2):
    """Get chat response with automatic retry for loading models"""
    for attempt in range(max_retries + 1):
        try:
            response = client.text_generation(
                model="google/flan-t5-large", 
                prompt=f"Answer concisely: {prompt}", 
                max_new_tokens=100
            )
            return response
        except Exception as e:
            error_msg = str(e).lower()
            # Check if model is loading/waking up
            if ("loading" in error_msg or "currently loading" in error_msg or 
                "503" in error_msg or "unavailable" in error_msg):
                if attempt < max_retries:
                    time.sleep(30)  # Wait 30 seconds before retry
                    continue
                return "⏳ The AI model is still loading. Please wait a moment and try again."
            else:
                return f"Error: {repr(e)}"
    return "⏳ The AI model is still loading. Please try again."

def generate_image(prompt, max_retries=2):
    """Generate image with automatic retry for loading models"""
    for attempt in range(max_retries + 1):
        try:
            image = client.text_to_image(
                prompt=prompt, 
                model="runwayml/stable-diffusion-v1-5"
            )
            return image
        except Exception as e:
            error_msg = str(e).lower()
            # Check if model is loading/waking up
            if ("loading" in error_msg or "currently loading" in error_msg or 
                "503" in error_msg or "unavailable" in error_msg):
                if attempt < max_retries:
                    time.sleep(30)  # Wait 30 seconds before retry
                    continue
                st.error("⏳ The Image model is still loading. Please try again in a moment.")
            else:
                st.error(f"Error: {repr(e)}")
            return None
    st.error("⏳ The Image model is still loading. Please try again.")
    return None

# --- UI DESIGN ---
st.set_page_config(page_title="Lab 10 AI", page_icon="🤖")
st.title("🤖 Multi-Modal AI Application")
st.caption("Lab 10.0 - Natural Language Processing (Large Language Models)")
st.divider()

tab1, tab2 = st.tabs(["💬 Chat", "🖼️ Image Generator"])

with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Thinking... (may take 30s if model is waking up)"):
            bot_reply = get_chat_response(user_input)
        
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

with tab2:
    img_prompt = st.text_area(
        "Describe the image you want to create:", 
        height=100, 
        placeholder="e.g., A cute cat astronaut in space"
    )
    
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if img_prompt:
            with st.spinner("Generating image... (may take 30s if model is waking up)"):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
