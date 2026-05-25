import streamlit as st
import requests
from PIL import Image
import io

HF_TOKEN = "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"

def get_chat_response(prompt):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        # 120 seconds timeout para hindi sya agad mag-surrender kung naglo-load
        response = requests.post(url, headers=headers, json={"inputs": f"Answer concisely: {prompt}"}, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            # Kunin ang generated text
            if isinstance(data, list) and 'generated_text' in data[0]:
                return data[0]['generated_text']
            return str(data)
        elif response.status_code == 503:
            # Naglo-load pa ang model
            return "⏳ The AI model is currently waking up. Please wait 30 seconds and type 'hello' again."
        else:
            return f"Error: {response.text}"
    except requests.exceptions.Timeout:
        return "⏳ Connection timed out. The model might be waking up. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image(prompt):
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        elif response.status_code == 503:
            st.error("⏳ The Image model is waking up. Please wait 30 seconds and click Generate again.")
            return None
        else:
            st.error(f"Error: {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏳ Connection timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
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
    img_prompt = st.text_area("Describe the image you want to create:", height=100, placeholder="e.g., A cute cat astronaut in space")
    if st.button("🎨 Generate Image", use_container_width=True, type="primary"):
        if img_prompt:
            with st.spinner("Generating image..."):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
