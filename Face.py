# Import necessary libraries
import streamlit as st
import face_recognition
import numpy as np
import os
from PIL import Image

# Function to load images and encode faces
def load_and_encode_images(image_folder):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    
    return known_face_encodings, known_face_names

# Load known faces
known_face_encodings, known_face_names = load_and_encode_images("known_faces")

# Streamlit interface
st.title("Face Recognition System")
st.write("Upload an image to recognize faces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Find all face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    
    # Recognize faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        st.write(f"Recognized: {name}")
