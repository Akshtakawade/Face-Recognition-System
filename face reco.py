import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
def load_image(image_path):
    return face_recognition.load_image_file(image_path)

def recognize_faces(known_face_encodings, known_face_names, unknown_image):
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)
    results = []
    
    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        results.append(name)
    
    return results
st.title("Face Recognition App")

# Upload known faces
uploaded_files = st.file_uploader("Upload Known Faces", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Process known faces
known_face_encodings = []
known_face_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = load_image(uploaded_file)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(uploaded_file.name)

# Upload unknown image
unknown_file = st.file_uploader("Upload Image to Recognize", type=["jpg", "jpeg", "png"])

if unknown_file:
    unknown_image = load_image(unknown_file)
    results = recognize_faces(known_face_encodings, known_face_names, unknown_image)
    st.image(unknown_image, caption="Uploaded Image", use_column_width=True)
    st.write("Recognized Faces: ", results)