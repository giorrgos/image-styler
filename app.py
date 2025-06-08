import streamlit as st
import replicate
import os
from dotenv import load_dotenv
from logger_config import setup_logger # Import the setup function

# --- Logger Setup ---
# Get the configured logger from logger_config.py
logger = setup_logger()

# --- Environment and Model Setup ---
# Load .env for Replicate API token
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Set Replicate model
MODEL = "black-forest-labs/flux-kontext-pro"

# Configure Replicate (Basic check for token)
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN not found in .env file or environment variables.")
    st.error("REPLICATE_API_TOKEN not found. Please set it in your .env file.")
    st.stop()

try:
    replicate.Client(api_token=REPLICATE_API_TOKEN)
    logger.info("Replicate client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Replicate client: {e}", exc_info=True)
    st.error(f"Failed to initialize Replicate client: {e}")
    st.stop()

def generate_styled_image_logic(ui_prompt, ui_uploaded_file, target_column_for_image):
    """
    Handles the logic for generating and displaying the styled image.
    The header for the styled image is expected to be placed in the target_column outside this function.
    """
    
    if not ui_uploaded_file:
        logger.warning("Image generation attempted without an uploaded file.")
        st.warning("Please upload an image first.")
        return
    if not ui_prompt:
        logger.warning("Image generation attempted without a prompt.")
        st.warning("Please enter a prompt.")
        return

    with st.spinner("Generating your image..."):
        try:
            # Ensure the file pointer is at the beginning for replicate
            ui_uploaded_file.seek(0)
            # Log model and prompt details
            logger.info(f"Calling Replicate API. Model: {MODEL}, Prompt: '{ui_prompt}'")
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
                # Log e) success if an image is returned and add the url
                logger.info(f"Successfully generated styled image. URL: {styled_image_url}")
                with target_column_for_image:
                    st.image(styled_image_url, caption="Styled Image", use_container_width=True)
            else:
                # Log if the call to replicate is unsuccessful (in terms of expected output)
                logger.warning(f"Image generation did not return a valid URL. API output: {api_output}")
                with target_column_for_image:
                    if api_output:
                        st.error(f"Image generation returned an unexpected result. Output: {api_output}")
                    else:
                        st.error("Image generation failed and no output was received.")

        except replicate.exceptions.ReplicateError as e:
            # Log if the call to replicate is unsuccessful (API error)
            logger.error(f"Replicate API call unsuccessful: {e}", exc_info=True)
            st.error(f"Replicate API Error: {e}")
        except Exception as e:
            # Log if the call to replicate is unsuccessful (other error)
            logger.error(f"An unexpected error occurred during image generation: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")

# --- Streamlit app setup ---
# Log the application starts
logger.info("Application started")
st.set_page_config(page_title="Image Styler", layout="wide")
st.title("Giorgos' Styling Salon")

# --- File uploader at the top ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="uploader")

# Corrected logic for logging image upload/removal
if uploaded_file is not None:
    # A file is currently uploaded
    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        # Log the image is uploaded (log only on new upload or change)
        size_in_bytes = uploaded_file.size
        size_in_mb = size_in_bytes / (1024 * 1024) # Convert bytes to MB
        logger.info(f"Image uploaded: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {size_in_mb:.2f} MB")
        st.session_state.last_uploaded_filename = uploaded_file.name
    # If 'last_uploaded_filename' exists and is the same as uploaded_file.name, no new log needed.
elif uploaded_file is None:
    # No file is currently uploaded
    if 'last_uploaded_filename' in st.session_state:
        # This means a file was just removed (or cleared)
        logger.info(f"Uploaded image removed: {st.session_state.last_uploaded_filename}") # Corrected this line
        del st.session_state.last_uploaded_filename


# --- Create two columns for image display ---
col1, col2 = st.columns(2)

# --- Original image display in the first column ---
with col1:
    st.header("Original Image")
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# --- Styled image display area in the second column ---
with col2:
    st.header("Styled Image")

# --- Prompt input below the images ---
prompt = st.text_input("Enter a style prompt (e.g. 'cyberpunk style portrait', 'ghibli studio')")

# --- "Generate Styled Image" button below the prompt ---
if st.button("Generate Styled Image"):
    generate_styled_image_logic(prompt, uploaded_file, col2)
