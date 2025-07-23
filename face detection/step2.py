import face_recognition
import cv2
import os

# Step 1: Load images of people you want to recognize
known_faces = []
known_names = []

directory = "known_people"  # Folder with known faces (e.g., Elon.jpg, Emma.jpg)

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Check if a face was found
            known_faces.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])  # Name from file
