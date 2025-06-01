import replicate
import os
from dotenv import load_dotenv

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- Step 2: Define Model and Inputs ---
# Define the Replicate model you want to use.
# Define the path to your input image and the style prompt.
print("\nStep 2: Defining model and inputs...")
MODEL_NAME = "black-forest-labs/flux-kontext-pro"
# IMPORTANT: Replace with the actual path to your image file
IMAGE_PATH = "giorrgos.jpg"
STYLE_PROMPT = "a vibrant oil painting"

# Initialize Replicate Client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Prepare Image File for API ---
# The Replicate API expects the image as a file object.
# We open the image file in binary read mode ("rb").
try:
    input_image_file = open(IMAGE_PATH, "rb")
    print(f"Image file '{IMAGE_PATH}' opened successfully.")
except FileNotFoundError:
    print(f"Error: Could not open image file at '{IMAGE_PATH}'. File not found.")
    exit()
except Exception as e:
    print(f"Error opening image file: {e}")
    exit()

# --- Step 5: Call Replicate API to Generate Styled Image ---
# This is where the main processing happens.
# We call the `replicate.run()` method with the model name and a dictionary of inputs.
# The inputs include the prompt and the opened image file.
print("\nStep 5: Calling Replicate API (this may take a moment)...")
api_output = None
try:
    api_output = client.run(
        MODEL_NAME,
        input={
            "prompt": STYLE_PROMPT,
            "input_image": input_image_file,
        }
    )
    print("API call successful.")
except replicate.exceptions.ReplicateError as e:
    print(f"Replicate API Error: {e}")
    if "free plan" in str(e).lower() or "credits" in str(e).lower():
        print("This might be due to exceeding free plan limits or insufficient credits.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during the API call: {e}")
    exit()
finally:
    # Always close the file after use
    if 'input_image_file' in locals() and not input_image_file.closed:
        input_image_file.close()
        print("Input image file closed.")

print(api_output)

# --- Step 6: Process API Output ---
# The API output needs to be inspected to get the URL of the styled image.
# Based on previous observations, the output might be a direct string URL
# or a list containing the URL.
print("\nStep 6: Processing API output...")
print(f"  Raw API Output: {api_output}")

styled_image_url = None
if api_output:
    try:
        # Attempt to convert the api_output to a string
        url_candidate = str(api_output)
        if url_candidate.startswith("http"):
            styled_image_url = url_candidate
            print(f"  Successfully extracted URL: {styled_image_url}")
            print(f"  Original API output type: {type(api_output)}")
        else:
            print(f"  Converted API output to string '{url_candidate}', but it does not appear to be a valid URL.")
            print(f"  Original API output type: {type(api_output)}")
    except Exception as e:
        print(f"  Error converting API output (type: {type(api_output)}) to string or processing it: {e}")
else:
    print("  API output is empty or None.")

# --- Step 7: Display Result ---
print("\nStep 7: Displaying result...")
if styled_image_url:
    print(f"  Successfully generated styled image!")
    print(f"  Styled Image URL: {styled_image_url}")
    print("  You can open this URL in a web browser to view the image.")
else:
    print("  Failed to generate or retrieve the styled image URL from the API response.")

print("\n--- Script Finished ---")