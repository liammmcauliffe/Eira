import face_recognition
import cv2
import numpy as np
from pathlib import Path

def load_known_faces():
    """Load known face encodings from the 'known_faces' directory"""
    known_encodings = []
    known_names = []
    
    known_faces_dir = Path("known_faces")
    if not known_faces_dir.exists():
        known_faces_dir.mkdir()
        return known_encodings, known_names
    
    for img_path in known_faces_dir.glob("*.jpg"):
        img = face_recognition.load_image_file(str(img_path))
        encodings = face_recognition.face_encodings(img)
        
        if encodings:
            known_encodings.append(encodings[0])
            name = img_path.stem.replace("_", " ").title()
            known_names.append(name)
    
    return known_encodings, known_names

def main():
    # Load known faces
    known_encodings, known_names = load_known_faces()
    
    if not known_encodings:
        print("No known faces loaded. Add reference images to 'known_faces/' directory")
        return
    
    # Initialize webcam
    video_capture = None
    for i in range(3):  # Try fewer camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            video_capture = cap
            break
        cap.release()
    
    if video_capture is None:
        print("Could not find working webcam")
        return
    
    # Process every other frame to improve performance
    process_this_frame = True
    
    while True:
        # Grab a single frame
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Process every other frame for better performance
        if process_this_frame:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            face_relationships = []
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.7)
                name = "Unknown"
                relationship = ""
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = known_names[best_match_idx]
                        # Set relationship based on name
                        if name.lower() == "liam mcauliffe":
                            relationship = "You"
                        else:
                            relationship = "Friend"
                
                face_names.append(name)
                face_relationships.append(relationship)
        
        process_this_frame = not process_this_frame
        
        # Display results on the full-size frame
        for (top, right, bottom, left), name, relationship in zip(face_locations, face_names, face_relationships):
            # Scale back up face locations (we processed at 1/4 size)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            
            # Get text sizes for centering
            (name_width, name_height), _ = cv2.getTextSize(name, font, font_scale, 1)
            name_x = left + (right - left - name_width) // 2
            name_y = top - 10
            
            # Draw transparent black background for name
            cv2.rectangle(frame, (name_x - 5, name_y - name_height - 5), 
                         (name_x + name_width + 5, name_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, name, (name_x, name_y), font, font_scale, (255, 255, 255), 1)
            
            if relationship:
                (rel_width, rel_height), _ = cv2.getTextSize(relationship, font, font_scale - 0.1, 1)
                rel_x = left + (right - left - rel_width) // 2
                rel_y = bottom + 20
                
                # Draw transparent black background for relationship
                cv2.rectangle(frame, (rel_x - 5, rel_y - rel_height - 5), 
                             (rel_x + rel_width + 5, rel_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, relationship, (rel_x, rel_y), font, font_scale - 0.1, (255, 255, 255), 1)
        
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()