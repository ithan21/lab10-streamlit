import streamlit as st
import time
from huggingface_hub import InferenceClient

HF_TOKEN = "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"

client = InferenceClient(token=HF_TOKEN, timeout=120)

def is_model_loading(error):
    """Check if error is because model is loading/waking up"""
    msg = str(error).lower()
    return any(keyword in msg for keyword in [
        "loading", "503", "unavailable", "wake", "sleep", 
        "stopiteration", "currently loaded", "warm"
    ])

def get_chat_response(prompt):
    max_retries = 6  # Try up to 6 times (3 minutes total)
    
    for attempt in range(max_retries):
        try:
            response = client.text_generation(
                model="google/flan-t5-large", 
                prompt=f"Answer concisely: {prompt}", 
                max_new_tokens=100,
                do_sample=False
            )
            if response and response.strip():
                return response
            return "I couldn't generate a response. Please try again."
            
        except Exception as e:
            if is_model_loading(e) and attempt < max_retries - 1:
                # Show waiting progress
                wait_sec = 30
                progress = st.progress(0, text=f"⏳ Model is waking up... (attempt {attempt+1}/{max_retries})")
                for i in range(wait_sec):
                    time.sleep(1)
                    progress.progress((i + 1) / wait_sec, text=f"⏳ Waking up model... {wait_sec - i}s remaining")
                progress.empty()
                continue  # Retry
            else:
                return f"Error: Model took too long to wake up. Please try again later."

def generate_image(prompt):
    max_retries = 6
    
    for attempt in range(max_retries):
        try:
            image = client.text_to_image(
                prompt=prompt, 
                model="runwayml/stable-diffusion-v1-5"
            )
            return image
            
        except Exception as e:
            if is_model_loading(e) and attempt < max_retries - 1:
                wait_sec = 30
                progress = st.progress(0, text=f"⏳ Image model is waking up... (attempt {attempt+1}/{max_retries})")
                for i in range(wait_sec):
                    time.sleep(1)
                    progress.progress((i + 1) / wait_sec, text=f"⏳ Waking up model... {wait_sec - i}s remaining")
                progress.empty()
                continue
            else:
                st.error("❌ Model took too long. Please try again later.")
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
        
        with st.spinner("Thinking..."):
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
            with st.spinner("Starting..."):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
