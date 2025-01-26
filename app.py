import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import tempfile
import shutil
import openpyxl
from model import model_predict

# Set up the app
st.set_page_config(layout="wide")

# Banner
st.header("Floor Plan Analyzer")
st.subheader("Upload a floor plan image and analyze its specifications")

# Header
uploaded_image = None
infer_button = st.button("Infer")

# Image upload

temp_file_path = None
uploaded_files = st.file_uploader("Choose a floor plan image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

spec_df = None
marked_image = None

uploaded_image_names = []
uploaded_images = []

subdir = 'uploads'
os.makedirs(subdir, exist_ok=True)

def save_to_subdir(images, names):

    #temp_dir = tempfile.TemporaryDirectory(dir = os.getcwd())
    #saved_images[temp_dir.name] = {}

    for img, name in zip(images, names):
        img_path = os.path.join(subdir, name)
        img.save(img_path)
        #saved_images[temp_dir.name][f"floor_plan_{i}"] = img_path

if uploaded_files is not None:
    for file in uploaded_files:
        uploaded_images.append(Image.open(file))
        uploaded_image_names.append(file.name)
    save_to_subdir(uploaded_images, uploaded_image_names)
    #uploaded_image.save(uploaded_image_name)

    #print(uploaded_image_names)
    # save_path = os.path.join(os.getcwd(), uploaded_image_name)
    # # Save the uploaded file to the specified location
    # with open(save_path, "wb") as output_file:
    #     output_file.write(uploaded_file.read())
    col1, col2 = st.columns([1, 1])

    with col1:
        if uploaded_images:
            image_placeholder = st.empty()
            image_placeholder.image(uploaded_images[0], caption="Uploaded Floor Plan", use_container_width=True)

    with col2:
        df_placeholder = st.empty()


# Placeholder for inference logic
def infer_floor_plan(image):
    # This is where you'd implement your actual inference logic
    # Return a DataFrame with room specifications and a marked image
    df, marked_image = model_predict(os.path.join(subdir, image))
    #df = pd.DataFrame({'Room': ['Living Room', 'Kitchen', 'Bedroom'],
    #                   'Area (sq.m)': [20, 10, 15]})
    #marked_image = image.copy()
    return df, marked_image


def analyze_floor_plans(images):
    all_results = []

    my_bar = st.progress(0, text="analysing images...")
    for k, img in enumerate(images):
        my_bar.progress((k+1)/len(images), text="analysing images...")
        try:
            # Perform analysis (placeholder function)
            df, marked_img = infer_floor_plan(img)

            all_results.append((df, marked_img))
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    my_bar.empty()
    return all_results


if infer_button and uploaded_images:
    with st.spinner('Analyzing floor plan...'):
        try:
            results = analyze_floor_plans(uploaded_image_names)
            spec_df = pd.concat([df for df, _ in results], ignore_index=True)
            #spec_df, marked_image = infer_floor_plan(uploaded_image_names)
            image_placeholder.image(results[0][1], caption=f"Analyzed Floor Plan for {uploaded_image_names[0]}", use_container_width=True)
        except:
            st.error("An error occurred while analyzing the floor plan.")

# Main content
if spec_df is not None:
    print(spec_df)
    df_placeholder.dataframe(spec_df, use_container_width=True, hide_index = True)

# Footer with download button
if spec_df is not None:
    @st.cache_data(ttl=3600)
    def convert_to_csv(df):
        return df.to_csv(index=False)

    csv = convert_to_csv(spec_df)

    @st.cache_data(ttl=3600)
    def convert_to_excel(df):
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        return buffer


    xlsx = convert_to_excel(spec_df)


    st.download_button(
        label="Download Specifications as CSV",
        data=csv,
        file_name="floor_plan_specifications.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Combined Specifications as XLSX",
        data=xlsx.getvalue(),
        file_name="combined_floor_plan_specifications.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )