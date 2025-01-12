import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Streamlit UI setup
st.set_page_config(page_title="Craiyon GenAI", layout="wide")
st.title("🎨 Craiyon (DALL·E Mini) Text-to-Image Generator")
st.markdown(
    """
    Generate images from text prompts using **Craiyon (formerly DALL·E Mini)**!  
    Simply enter your text prompt, and Craiyon will generate 9 unique images for you.
    """
)

# Input for text prompt
prompt = st.text_input(
    "Enter your text prompt:",
    value="A futuristic cityscape at night, neon lights, cyberpunk style"
)

# Button to trigger generation
if st.button("Generate"):
    with st.spinner("Generating images... This might take up to a minute."):
        try:
            # Send the request to Craiyon API
            response = requests.post(
                "https://backend.craiyon.com/generate",  # Craiyon backend API
                json={"prompt": prompt},
                timeout=60  # Timeout in case of delays
            )
            response.raise_for_status()

            # Extract image data from response
            image_data = response.json().get("images", [])
            if not image_data:
                st.error("No images were generated. Try a different prompt!")
            else:
                st.success("Images generated successfully!")
                st.markdown("### Your Generated Images:")

                # Display images in a grid
                cols = st.columns(3)
                for i, img_hex in enumerate(image_data):
                    img = Image.open(BytesIO(bytes.fromhex(img_hex)))
                    cols[i % 3].image(img, caption=f"Image {i + 1}", use_column_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
