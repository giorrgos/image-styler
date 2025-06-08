import streamlit as st
import replicate
import os
from dotenv import load_dotenv

# Load .env for Replicate API token
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Set Replicate model
MODEL = "black-forest-labs/flux-kontext-pro"

# Configure Replicate (Basic check for token)
if not REPLICATE_API_TOKEN:
    st.error("REPLICATE_API_TOKEN not found. Please set it in your .env file.")
    st.stop()

try:
    replicate.Client(api_token=REPLICATE_API_TOKEN)
except Exception as e:
    st.error(f"Failed to initialize Replicate client: {e}")
    st.stop()

def generate_styled_image_logic(ui_prompt, ui_uploaded_file, target_column_for_image):
    """
    Handles the logic for generating and displaying the styled image.
    The header for the styled image is expected to be placed in the target_column outside this function.
    """
    if not ui_uploaded_file:
        st.warning("Please upload an image first.")
        return
    if not ui_prompt:
        st.warning("Please enter a prompt.")
        return

    with st.spinner("Generating your image..."):
        try:
            # Ensure the file pointer is at the beginning for replicate
            ui_uploaded_file.seek(0)
            api_output = replicate.run(
                MODEL,
                input={
                    "prompt": ui_prompt,
                    "input_image": ui_uploaded_file,
                }
            )

            styled_image_url = None
            if api_output:
                url_candidate = str(api_output)
                if url_candidate.startswith("http"):
                    styled_image_url = url_candidate

            if styled_image_url:
                with target_column_for_image: # Use the passed column object
                    # st.header("Styled Image") # Header is now placed outside this function
                    st.image(styled_image_url, caption="Styled Image", use_container_width=True)
            else:
                # Display errors within the target column if image generation fails at this stage
                with target_column_for_image:
                    if api_output:
                        st.error(f"Image generation returned an unexpected result. Output: {api_output}")
                    else:
                        st.error("Image generation failed and no output was received.")

        except replicate.exceptions.ReplicateError as e:
            # These errors will appear globally or can be directed to target_column_for_image if preferred
            st.error(f"Replicate API Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Streamlit app setup ---
st.set_page_config(page_title="Image Styler", layout="wide")
st.title("Giorgos' Styling Salon")

# --- File uploader at the top ---
# Moved here so it doesn't interfere with column alignment for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="uploader")

# --- Create two columns for image display ---
col1, col2 = st.columns(2)

# --- Original image display in the first column ---
with col1:
    st.header("Original Image")
    # Image is displayed directly under the header if a file is uploaded
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# --- Styled image display area in the second column ---
with col2:
    st.header("Styled Image") # Header is always visible here
    # The image itself will be added by generate_styled_image_logic into this column (col2)

# --- Prompt input below the images ---
prompt = st.text_input("Enter a style prompt (e.g. 'cyberpunk style portrait', 'ghibli studio')")

# --- "Generate Styled Image" button below the prompt ---
if st.button("Generate Styled Image"):
    # Pass col2, where the image (not header) should be placed by the function
    generate_styled_image_logic(prompt, uploaded_file, col2)