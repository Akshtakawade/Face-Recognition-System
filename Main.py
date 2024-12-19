import cv2
import face_recognition
import numpy as np
from Image_Encoding import known_face_encodings,known_face_names

# print(known_face_encodings)
# print(known_face_names)

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # 0 for the first webcam


while True: 
    # Grab a frame from the video
    ret, frame = video_capture.read()
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    # rgb_frame = frame[:, :, ::-1]
    print(frame.shape)
    # Find all faces and their encodings
    face_locations = face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame)
    
    # Loop through each detected face
    for (top, right, bottom, left), face_encod in zip(face_locations, face_encoding):
        # Check if the face matches any known face encoding
        matches = face_recognition.compare_faces(known_face_encodings, face_encod)
        print(f"matches : {matches}")
        # Set a default name if no match is found
        name = "Unknown"
        
        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encod)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Draw a box around the face and label it with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0, 255), 2)
        cv2.rectangle(frame, ( left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
