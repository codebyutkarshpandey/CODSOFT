# Step 2: Start webcam
video = cv2.VideoCapture(0)

print("📷 Starting camera... Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Step 3: Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Step 4: Use the closest match
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if matches:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale back up since we resized earlier
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle & name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
