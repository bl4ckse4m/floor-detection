import streamlit as st
import pandas as pd
from PIL import Image
import io
import os

# Set up the app
st.set_page_config(layout="wide")

# Banner
st.header("Floor Plan Analyzer")
st.subheader("Upload a floor plan image and analyze its specifications")

# Header
uploaded_image = None
infer_button = st.button("Infer")

# Image upload
uploaded_file = st.file_uploader("Choose a floor plan image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Floor Plan")

# Placeholder for inference logic
def infer_floor_plan(image):
    # This is where you'd implement your actual inference logic
    # Return a DataFrame with room specifications and a marked image
    df = pd.DataFrame({'Room': ['Living Room', 'Kitchen', 'Bedroom'],
                       'Area (sq.m)': [20, 10, 15]})
    marked_image = image.copy()
    return df, marked_image

spec_df = None
marked_image = None

if infer_button and uploaded_image:
    spec_df, marked_image = infer_floor_plan(uploaded_image)
    st.image(marked_image, caption="Analyzed Floor Plan")

# Main content
if spec_df is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(marked_image, caption="Analyzed Floor Plan")

    with col2:
        st.dataframe(spec_df)

# Footer with download button
if spec_df is not None:
    @st.cache_data(ttl=3600)
    def convert_df(df):
        return df.to_csv(index=False)

    csv = convert_df(spec_df)

    st.download_button(
        label="Download Specifications as CSV",
        data=csv,
        file_name="floor_plan_specifications.csv",
        mime="text/csv",
    )
