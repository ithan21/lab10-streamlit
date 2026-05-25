import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import time

# Direktang token
HF_TOKEN = "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"

# Gumagamit ng official HF client
client = InferenceClient(token=HF_TOKEN)

# --- FUNCTIONS ---
def get_chat_response(prompt):
    """May auto-retry para sa paggising ng model"""
    full_prompt = f"Please answer concisely: {prompt}"
    
    # Subukan nang 3 beses
    for attempt in range(3):
        try:
            response = client.text_generation(
                model="google/flan-t5-large",
                prompt=full_prompt,
                max_new_tokens=100
            )
            return response
        except Exception as e:
            if attempt < 2:
                time.sleep(15) # Hintay 15 seconds bago subukan ulit
            else:
                return f"⏳ The AI model is taking too long to wake up. Please type 'hello' again."

def generate_image(prompt):
    """May auto-retry para sa paggising ng image model"""
    
    # Subukan nang 3 beses
    for attempt in range(3):
        try:
            image = client.text_to_image(
                prompt=prompt,
                model="stabilityai/stable-diffusion-xl-base-1.0"
            )
            return image
        except Exception as e:
            if attempt < 2:
                time.sleep(20) # Hintay 20 seconds bago subukan ulit
            else:
                st.error("⏳ The Image model is taking too long to wake up. Please click Generate again.")
                return None

# --- UI DESIGN ---
st.set_page_config(page_title="Lab 10 AI", page_icon="🤖")

st.title("🤖 Multi-Modal AI Application")
st.caption("Lab 10.0 - Natural Language Processing (Large Language Models)")
st.divider()

tab1, tab2 = st.tabs(["💬 Chat", "🖼️ Image Generator"])

# --- TAB 1: CHAT ---
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

        with st.spinner("Thinking... (If model is asleep, it will auto-retry for 45 seconds)"):
            bot_reply = get_chat_response(user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

# --- TAB 2: IMAGE GENERATOR ---
with tab2:
    img_prompt = st.text_area("Describe the image you want to create:", height=100, placeholder="e.g., A cute cat astronaut in space, digital art")
    
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if img_prompt:
            with st.spinner("Generating image... (Auto-retrying if model is asleep)"):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
