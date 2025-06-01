import streamlit as st
import replicate
import os
from dotenv import load_dotenv
from PIL import Image

# Load .env for Replicate API token
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Set Replicate model
MODEL = "black-forest-labs/flux-kontext-pro"

# Configure Replicate
if REPLICATE_API_TOKEN:
    # Initialize the client, but we will use replicate.run for simplicity in Streamlit
    # The client initialization here mainly serves to check if the token is set.
    try:
        replicate.Client(api_token=REPLICATE_API_TOKEN)
    except Exception as e:
        st.error(f"Failed to initialize Replicate client: {e}")
        st.stop()
else:
    st.error("REPLICATE_API_TOKEN not found. Please set it in your .env file.")
    st.stop()

st.set_page_config(page_title="Flux Image Styler", layout="wide")
st.title("🖼️ Giorgos' Styling Place")

# Create two columns for image display
col1, col2 = st.columns(2)

with col1:
    st.header("Original Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="uploader")
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

prompt = st.text_input("Enter a style prompt (e.g. 'cyberpunk style portrait')")

if st.button("Generate Styled Image"):
    if not uploaded_file:
        st.warning("Please upload an image first.")
    elif not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                # Ensure the file pointer is at the beginning for replicate
                uploaded_file.seek(0)
                api_output = replicate.run(
                    MODEL,
                    input={
                        "prompt": prompt,
                        "input_image": uploaded_file,
                    }
                )

                st.write("Debug: Raw API Output from Replicate:", api_output) # Keep for debugging

                styled_image_url = None
                if api_output:
                    # Attempt to convert the api_output to a string, similar to debug.py
                    url_candidate = str(api_output)
                    if url_candidate.startswith("http"):
                        styled_image_url = url_candidate
                    # Fallback for list, though string conversion should handle most cases
                    elif isinstance(api_output, list) and len(api_output) > 0 and isinstance(api_output[0], str) and api_output[0].startswith("http"):
                        styled_image_url = api_output[0]

                if styled_image_url:
                    with col2:
                        st.header("Styled Image")
                        st.image(styled_image_url, caption="Styled Image", use_container_width=True)
                else:
                    st.error(f"Image generation failed or returned an unexpected result. Output: {api_output}")
            except replicate.exceptions.ReplicateError as e:
                st.error(f"Replicate API Error: {e}")
                if "free plan" in str(e).lower() or "credits" in str(e).lower():
                    st.warning("This might be due to exceeding free plan limits or insufficient credits with Replicate.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")