import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Garden Image Processor", layout="wide")

# Title
st.title("Garden Image - Multi-Color Channel Visualizer")

# Load image from URL
@st.cache_data
def load_image():
    url = r"C:\Users\Ramya\OneDrive\Desktop\garden img.avif"
    return Image.open(url).convert("RGB")

# ✅ CORRECT VARIABLE NAME
garden = load_image()

# Display image
st.image(garden, caption="Original Garden Image", use_container_width=True)

# Convert to NumPy array
garden_np = np.array(garden)
R, G, B = garden_np[:, :, 0], garden_np[:, :, 1], garden_np[:, :, 2]

# Create channel images
red_img = np.zeros_like(garden_np)
green_img = np.zeros_like(garden_np)
blue_img = np.zeros_like(garden_np)

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

garden_gray = garden.convert("L")
garden_gray_np = np.array(garden_gray)

# Plot using matplotlib with colormap
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(garden_gray_np, cmap=colormap)
ax.axis("off")

st.pyplot(fig)
