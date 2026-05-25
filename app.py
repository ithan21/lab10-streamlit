import streamlit as st
import requests
import urllib.parse
from groq import Groq
from PIL import Image
import io

# ==========================================
# STEP 4: ILAGAY MO ANG COPIED API KEY DITO
# ==========================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)

def get_chat_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {repr(e)}"

def generate_image(prompt):
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        # Pollinations.ai - Libre, walang API key, walang tulog!
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"
        
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            return Image.open(io.BytesIO(img_response.content))
        else:
            st.error("Failed to generate image.")
            return None
    except Exception as e:
        st.error(f"Error: {repr(e)}")
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
            with st.spinner("Generating image..."):
                img = generate_image(img_prompt)
                if img:
                    st.image(img, caption=img_prompt, use_column_width=True)
        else:
            st.warning("Please type a description first.")

st.divider()
st.caption("Made with Streamlit | Chat powered by Groq | Image powered by Pollinations.ai")
