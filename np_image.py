import streamlit as st 
import numpy as np 
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Dog Image Processor", layout="wide")

# Title
st.title("Dog Image - Multi-Color Channel Visualizer")

# Load image from URL with headers
@st.cache_data 
def load_image():
    url = "https://thumbs.dreamstime.com/b/golden-retriever-dog-21668976.jpg"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None

# Load and display image
dog = load_image()
if dog is None:
    st.stop()

st.image(dog, caption="Original Dog Image", use_container_width=True)

# Convert to NumPy array
dog_np = np.array(dog)
R, G, B = dog_np[:, :, 0], dog_np[:, :, 1], dog_np[:, :, 2]

# Create channel images
red_img = np.zeros_like(dog_np)
green_img = np.zeros_like(dog_np)
blue_img = np.zeros_like(dog_np)

red_img[:, :, 0] = R
green_img[:, :, 1] = G
blue_img[:, :, 2] = B

# Display RGB channels
st.subheader("RGB Channel Visualization")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(red_img, caption="Red Channel", use_container_width=True)

with col2:
    st.image(green_img, caption="Green Channel", use_container_width=True)

with col3:
    st.image(blue_img, caption="Blue Channel", use_container_width=True)

# Grayscale + Colormap
st.subheader("Colormapped Grayscale Image")

colormap = st.selectbox(
    "Choose a Matplotlib colormap",
    ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "cool", "gray"]
)

dog_gray = dog.convert("L")
dog_gray_np = np.array(dog_gray)

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(dog_gray_np, cmap=colormap)
ax.axis("off")

st.pyplot(fig)