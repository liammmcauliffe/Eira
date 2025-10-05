import face_recognition
import cv2
import numpy as np
from pathlib import Path
from supabase import create_client, Client
from datetime import datetime
import os
import pickle
import hashlib
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def init_supabase() -> Client:
    """Initialize Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_or_create_person(supabase: Client, filename: str, relationship: str):
    """Get existing person or create new one"""
    name = filename.replace('.jpg', '').replace('_', ' ').title()
    
    # Check if person exists
    result = supabase.table("persons").select("id").eq("name", name).execute()
    
    if result.data:
        return result.data[0]['id']
    else:
        # Create new person
        person_data = {
            "name": name,
            "relationship": relationship,
            "image_filename": filename
        }
        result = supabase.table("persons").insert(person_data).execute()
        return result.data[0]['id']

def get_cached_encoding(supabase: Client, person_id: int):
    """Get cached face encoding from database"""
    result = supabase.table("face_encodings").select("encoding_data").eq("person_id", person_id).execute()
    
    if result.data:
        # Deserialize the encoding (it's stored as base64)
        encoding_b64 = result.data[0]['encoding_data']
        encoding_bytes = base64.b64decode(encoding_b64)
        return pickle.loads(encoding_bytes)
    return None

def cache_encoding(supabase: Client, person_id: int, encoding: np.ndarray):
    """Cache face encoding in database"""
    # Serialize the encoding and encode as base64 for JSON compatibility
    encoding_bytes = pickle.dumps(encoding)
    encoding_b64 = base64.b64encode(encoding_bytes).decode('utf-8')
    
    encoding_data = {
        "person_id": person_id,
        "encoding_data": encoding_b64
    }
    
    # Delete old encoding if exists
    supabase.table("face_encodings").delete().eq("person_id", person_id).execute()
    
    # Insert new encoding
    supabase.table("face_encodings").insert(encoding_data).execute()

def load_known_faces(supabase: Client):
    """Load known face encodings with caching"""
    known_encodings = []
    known_names = []
    known_relationships = []
    
    try:
        # List files in the face-photos bucket
        files = supabase.storage.from_("face-photos").list()
        
        for file_info in files:
            if file_info['name'].endswith('.jpg'):
                filename = file_info['name']
                name = filename.replace('.jpg', '').replace('_', ' ').title()
                
                # Determine relationship
                if name.lower() == "liam mcauliffe":
                    relationship = "You"
                else:
                    relationship = "Friend"
                
                # Get or create person
                person_id = get_or_create_person(supabase, filename, relationship)
                
                # Try to get cached encoding
                encoding = get_cached_encoding(supabase, person_id)
                
                if encoding is None:
                    print(f"Processing new face: {name}")
                    # Download and process image
                    image_data = supabase.storage.from_("face-photos").download(filename)
                    
                    # Convert bytes to numpy array for face_recognition
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    encodings = face_recognition.face_encodings(img_rgb)
                    
                    if encodings:
                        encoding = encodings[0]
                        # Cache the encoding
                        cache_encoding(supabase, person_id, encoding)
                        print(f"Cached encoding for: {name}")
                    else:
                        print(f"No face found in: {filename}")
                        continue
                else:
                    print(f"Using cached encoding for: {name}")
                
                known_encodings.append(encoding)
                known_names.append(name)
                known_relationships.append(relationship)
    
    except Exception as e:
        print(f"Error loading faces: {e}")
    
    return known_encodings, known_names, known_relationships

def log_detection(supabase: Client, person_id: int):
    """Log face detection to Supabase"""
    try:
        data = {
            "person_id": person_id
        }
        supabase.table("face_detections").insert(data).execute()
    except Exception as e:
        print(f"Error logging to Supabase: {e}")

def main():
    # Initialize Supabase
    supabase = init_supabase()
    
    # Load known faces from Supabase storage with caching
    known_encodings, known_names, known_relationships = load_known_faces(supabase)
    
    if not known_encodings:
        print("No known faces loaded. Add reference images to 'face-photos' storage bucket")
        return
    
    # Initialize webcam
    video_capture = None
    for i in range(3):
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
    last_logged = {}  # Track last log time for each person
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            face_relationships = []
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.7)
                name = "Unknown"
                relationship = ""
                
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = known_names[best_match_idx]
                        relationship = known_relationships[best_match_idx]
                        
                        # Log to Supabase (only once every 30 seconds per person)
                        current_time = datetime.now()
                        if name not in last_logged or (current_time - last_logged[name]).seconds > 30:
                            # Get person_id for logging
                            person_result = supabase.table("persons").select("id").eq("name", name).execute()
                            if person_result.data:
                                log_detection(supabase, person_result.data[0]['id'])
                            last_logged[name] = current_time
                
                face_names.append(name)
                face_relationships.append(relationship)
        
        process_this_frame = not process_this_frame
        
        # Display results on the full-size frame
        for (top, right, bottom, left), name, relationship in zip(
            face_locations, face_names, face_relationships
        ):
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