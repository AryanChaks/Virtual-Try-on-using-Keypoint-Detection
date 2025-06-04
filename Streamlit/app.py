import streamlit as st
import cv2
import numpy as np
import datetime
import os
from dotenv import load_dotenv
import time
import json

from constants import IMAGE_DIR, SEGMENTATION_IMAGE_DIR, OPENPOSE_IMAGE_DIR, OPENPOSE_JSON_DIR, FINAL_OUTPUT_DIR, DATASET_DIR, CLOTH_DIR
from segmentation import get_segmentation_output
from pose_estimation import get_openpose_output 
from virtual_try_on import call_virtual_try_on

load_dotenv()

# Folder to save images
os.makedirs(IMAGE_DIR, exist_ok=True)

def read_gesture_data():
    """Read gesture data from JSON file and update session state"""
    try:
        with open(f'{os.environ.get("GESTURE_DATA_PATH")}/gesture_data.json', 'r') as f:
            gesture_data = json.load(f)

        # Check if the data is recent (within last 5 seconds)
        if time.time() - gesture_data['timestamp'] < 5:
            return {
                'hand_sign': gesture_data['hand_sign'],
                'finger_gesture': gesture_data['finger_gesture']
            }
        else:
            return {'hand_sign': None, 'finger_gesture': None}
    except (FileNotFoundError, json.JSONDecodeError):
        return {'hand_sign': None, 'finger_gesture': None}

# Initialize session states
if 'gesture_data' not in st.session_state:
    st.session_state.gesture_data = {'hand_sign': None, 'finger_gesture': None}
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

run_name = 'inference-003'

def main():
    st.title("📷 Virtual Try On App")

    # Add a checkbox to control gesture detection
    auto_refresh = st.checkbox('Enable Gesture Detection', value=True)

    # Image upload
    uploaded_file = st.file_uploader("Upload an image to try on virtual clothes", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"photo_{timestamp}.jpg"
        filename = os.path.join(IMAGE_DIR, img_filename)

        # Save as JPG
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        st.success(f"Photo saved as {filename}")

        # show clothes images
        clothes_images = os.listdir(CLOTH_DIR)
        clothes_images = {img: os.path.join(CLOTH_DIR, img) for img in clothes_images}
        clothes_list = list(clothes_images.keys())

        if auto_refresh and (time.time() - st.session_state.last_update) > 0.1:
            st.session_state.gesture_data = read_gesture_data()
            st.session_state.last_update = time.time()
            
            # Update selected index based on gesture
            if st.session_state.gesture_data['hand_sign'] == 'Pointer':
                if st.session_state.gesture_data['finger_gesture'] == 'Clockwise':
                    st.session_state.selected_index = (st.session_state.selected_index + 1) % len(clothes_list)
                elif st.session_state.gesture_data['finger_gesture'] is not None and st.session_state.gesture_data['finger_gesture'] == 'Counter Clockwise':
                    st.session_state.selected_index = (st.session_state.selected_index - 1) % len(clothes_list)

        # Use the updated index to select the cloth
        selected_cloth = clothes_list[st.session_state.selected_index]
        selected_cloth_image = clothes_images[selected_cloth]

        col1, col2 = st.columns(2)

        with col1:
            # Display uploaded image
            st.image(frame, channels="BGR", caption="Uploaded Image", use_container_width=True)

        with col2:
            # Create and use placeholder within the column context
            cloth_placeholder = st.empty()
            cloth_placeholder.image(selected_cloth_image, channels="BGR", caption=f"Selected Cloth: {selected_cloth}", use_container_width=True)

        if st.button("Try On"):
            # save the image filename in test_pairs.txt file
            with open(os.path.join(DATASET_DIR, f'{run_name}.txt'), 'a') as txt_file:
                txt_file.write(f"{img_filename} {selected_cloth}\n")

            steps = ["Segmentation", "Pose Detection", "Augmentation", "Completed"]
            progress_bar = st.progress(0)

            for i, step in enumerate(steps):
                with st.spinner(f"{step} in progress..."):
                    if step == "Segmentation":
                        get_segmentation_output(os.getenv('SEGMENTATION_CONDA_PATH'), os.getenv('SEGMENTATION_PATH'), os.path.abspath(filename), SEGMENTATION_IMAGE_DIR)

                    elif step == "Pose Detection":
                        get_openpose_output(os.getenv('OPENPOSE_PATH'), os.path.abspath(filename), OPENPOSE_IMAGE_DIR, OPENPOSE_JSON_DIR)

                    elif step == "Augmentation":
                        call_virtual_try_on(os.getenv('VITON_HD_PATH'), os.getenv('VITON_HD_VIRTUAL_ENV_PATH'), run_name, os.path.abspath(DATASET_DIR), FINAL_OUTPUT_DIR)

                progress_bar.progress((i + 1) / len(steps))

            output_image = os.path.join(FINAL_OUTPUT_DIR, run_name, f'photo_{selected_cloth}')
            st.image(output_image, channels="BGR", caption="Virtual Try On", use_container_width=True)

if __name__ == "__main__":
    main()

    