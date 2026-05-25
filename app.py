import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

HF_TOKEN = "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"

client = InferenceClient(token=HF_TOKEN, timeout=120)

def get_chat_response(prompt):
    try:
        response = client.text_generation(
            model="google/flan-t5-large", 
            prompt=f"Answer concisely: {prompt}", 
            max_new_tokens=100,
            do_sample=False  # This prevents StopIteration!
        )
        if not response or response.strip() == "":
            return "I couldn't generate a response. Please try rephrasing."
        return response
    except StopIteration:
        return "⏳ Model is waking up. Please wait 30 seconds and try again."
    except Exception as e:
        error_str = str(e).lower()
        if "loading" in error_str or "503" in error_str:
            return "⏳ Model is waking up. Please wait 30 seconds and try again."
        return f"Error: {repr(e)}"

def generate_image(prompt):
    try:
        image = client.text_to_image(
            prompt=prompt, 
            model="runwayml/stable-diffusion-v1-5"
        )
        return image
    except StopIteration:
        st.error("⏳ Model is waking up. Wait 30 seconds and try again.")
        return None
    except Exception as e:
        error_str = str(e).lower()
        if "loading" in error_str or "503" in error_str:
            st.error("⏳ Model is waking up. Wait 30 seconds and try again.")
        else:
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
st.caption("Made with Streamlit | AI Models powered by Hugging Face API")
