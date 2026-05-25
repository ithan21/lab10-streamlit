import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import time

HF_TOKEN = "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"

client = InferenceClient(token=HF_TOKEN, timeout=120)

def get_chat_response(prompt):
    full_prompt = f"Please answer concisely: {prompt}"
    for attempt in range(3):
        try:
            response = client.text_generation(model="google/flan-t5-large", prompt=full_prompt, max_new_tokens=100)
            return response
        except Exception as e:
            if attempt < 2:
                time.sleep(20)
            else:
                return f"Error: {str(e)}"

def generate_image(prompt):
    for attempt in range(3):
        try:
            image = client.text_to_image(prompt=prompt, model="runwayml/stable-diffusion-v1-5")
            return image
        except Exception as e:
            if attempt < 2:
                time.sleep(25)
            else:
                st.error(f"Error: {str(e)}")
                return None

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
        with st.spinner("⏳ Waking up AI... Please wait up to 1-2 minutes..."):
            bot_reply = get_chat_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

with tab2:
    img_prompt = st.text_area("Describe the image you want to create:", height=100, placeholder="e.g., A cute cat astronaut in space")
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if img_prompt:
            with st.spinner("⏳ Generating image... Please wait up to 1-2 minutes..."):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
