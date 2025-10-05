import face_recognition
import cv2
import numpy as np
from pathlib import Path
from supabase import create_client, Client
from datetime import datetime, timedelta
import os
import pickle
import hashlib
import base64
import time
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

def get_current_and_next_tasks(supabase: Client):
    """Get current and next tasks from database with 10-minute buffer"""
    try:
        # Get all active tasks ordered by time
        result = supabase.table("tasks").select("*").eq("is_active", True).order("scheduled_time").execute()
        
        if not result.data:
            return None, None
        
        now = datetime.now().time()
        current_task = None
        next_task = None
        
        # Add 10-minute buffer for task completion
        buffer_minutes = 10
        
        for task in result.data:
            task_time = datetime.strptime(task['scheduled_time'], '%H:%M:%S').time()
            
            # Calculate end time (task time + buffer)
            task_datetime = datetime.combine(datetime.today(), task_time)
            end_datetime = task_datetime + timedelta(minutes=buffer_minutes)
            end_time = end_datetime.time()
            
            # Check if we're within the task window (task_time to end_time)
            if task_time <= now <= end_time:
                current_task = task
            elif now < task_time:
                # This is a future task
                if next_task is None:
                    next_task = task
                break
        
        return current_task, next_task
        
    except Exception as e:
        print(f"Error getting tasks: {e}")
        return None, None

def calculate_time_until(target_time_str):
    """Calculate time until target time"""
    try:
        target_time = datetime.strptime(target_time_str, '%H:%M:%S').time()
        now = datetime.now().time()
        
        # Convert to datetime for calculation
        now_dt = datetime.combine(datetime.today(), now)
        target_dt = datetime.combine(datetime.today(), target_time)
        
        # If target time has passed today, it's for tomorrow
        if target_dt <= now_dt:
            target_dt = target_dt.replace(day=target_dt.day + 1)
        
        time_diff = target_dt - now_dt
        total_seconds = int(time_diff.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return hours, minutes, seconds
        
    except Exception as e:
        print(f"Error calculating time: {e}")
        return 0, 0, 0

def draw_taskbar(frame, current_task, next_task, taskbar_cache):
    """Draw beautiful taskbar at the top of the frame using cached data"""
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]
        taskbar_height = 80
        
        # Create gradient background (dark blue to darker blue)
        for y in range(taskbar_height):
            alpha = y / taskbar_height
            color_intensity = int(30 + alpha * 20)  # Gradient from dark to darker
            cv2.line(frame, (0, y), (width, y), (color_intensity, color_intensity + 10, color_intensity + 20), 1)
        
        # Add subtle border
        cv2.rectangle(frame, (0, 0), (width, taskbar_height), (100, 120, 140), 2)
        
        # Current time with better styling
        current_time = datetime.now().strftime("%I:%M %p")
        # Add background circle for time
        cv2.circle(frame, (50, 25), 20, (60, 80, 100), -1)
        cv2.circle(frame, (50, 25), 20, (120, 140, 160), 2)
        cv2.putText(frame, current_time, (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Task information with better styling
        if current_task:
            # Check if task is ending soon (within 5 minutes)
            task_time = datetime.strptime(current_task['scheduled_time'], '%H:%M:%S').time()
            task_datetime = datetime.combine(datetime.today(), task_time)
            end_datetime = task_datetime + timedelta(minutes=10)
            now_datetime = datetime.now()
            
            # Current task styling
            if now_datetime >= end_datetime - timedelta(minutes=5):
                task_text = f"ENDING SOON: {current_task['title']}"
                # Orange background for ending soon
                text_size = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (255, 165, 0), -1)  # Orange background
                cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (255, 200, 100), 2)  # Orange border
            else:
                task_text = f"NOW: {current_task['title']}"
                # Red background for current task
                text_size = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (220, 50, 50), -1)  # Red background
                cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (255, 100, 100), 2)  # Red border
            
            cv2.putText(frame, task_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif next_task:
            hours, minutes, seconds = calculate_time_until(next_task['scheduled_time'])
            task_text = f"NEXT: {next_task['title']}"
            
            # Background rectangle for next task
            text_size = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (50, 120, 200), -1)  # Blue background
            cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (100, 150, 255), 2)  # Blue border
            cv2.putText(frame, task_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Countdown with styling
            if hours > 0:
                countdown_text = f"in {hours}h {minutes}m"
            elif minutes > 0:
                countdown_text = f"in {minutes}m {seconds}s"
            else:
                countdown_text = f"in {seconds}s"
            
            # Countdown background
            countdown_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (width - countdown_size[0] - 25, 45), (width - 10, 70), (40, 40, 40), -1)
            cv2.rectangle(frame, (width - countdown_size[0] - 25, 45), (width - 10, 70), (100, 100, 100), 2)
            cv2.putText(frame, countdown_text, (width - countdown_size[0] - 20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            
        else:
            # No tasks styling
            no_tasks_text = "No more tasks today"
            text_size = cv2.getTextSize(no_tasks_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (15, 45), (25 + text_size[0], 70), (80, 80, 80), -1)
            cv2.putText(frame, no_tasks_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Add subtle bottom border
        cv2.line(frame, (0, taskbar_height - 2), (width, taskbar_height - 2), (120, 140, 160), 2)
        
    except Exception as e:
        print(f"Error drawing taskbar: {e}")

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
    
    # Taskbar caching for performance
    taskbar_cache = {
        'current_task': None,
        'next_task': None,
        'last_update': 0
    }
    
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
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            
            # Get text sizes for centering
            (name_width, name_height), _ = cv2.getTextSize(name, font, font_scale, 2)
            name_x = left + (right - left - name_width) // 2
            name_y = top - 15
            
            # Draw beautiful rounded background for name
            padding = 12
            cv2.rectangle(frame, (name_x - padding, name_y - name_height - padding), 
                         (name_x + name_width + padding, name_y + padding), (40, 40, 40), -1)
            cv2.rectangle(frame, (name_x - padding, name_y - name_height - padding), 
                         (name_x + name_width + padding, name_y + padding), (100, 150, 200), 3)
            
            # Add subtle shadow effect
            cv2.putText(frame, name, (name_x + 1, name_y + 1), font, font_scale, (0, 0, 0), 2)
            cv2.putText(frame, name, (name_x, name_y), font, font_scale, (255, 255, 255), 2)
            
            if relationship:
                (rel_width, rel_height), _ = cv2.getTextSize(relationship, font, font_scale - 0.2, 2)
                rel_x = left + (right - left - rel_width) // 2
                rel_y = bottom + 25
                
                # Draw beautiful background for relationship
                rel_padding = 10
                cv2.rectangle(frame, (rel_x - rel_padding, rel_y - rel_height - rel_padding), 
                             (rel_x + rel_width + rel_padding, rel_y + rel_padding), (60, 60, 60), -1)
                cv2.rectangle(frame, (rel_x - rel_padding, rel_y - rel_height - rel_padding), 
                             (rel_x + rel_width + rel_padding, rel_y + rel_padding), (150, 200, 100), 2)
                
                # Add subtle shadow effect
                cv2.putText(frame, relationship, (rel_x + 1, rel_y + 1), font, font_scale - 0.2, (0, 0, 0), 2)
                cv2.putText(frame, relationship, (rel_x, rel_y), font, font_scale - 0.2, (255, 255, 255), 2)
        
        # Update taskbar cache every 30 seconds for performance
        current_time = time.time()
        if current_time - taskbar_cache['last_update'] > 30:
            taskbar_cache['current_task'], taskbar_cache['next_task'] = get_current_and_next_tasks(supabase)
            taskbar_cache['last_update'] = current_time
        
        # Draw taskbar at the top using cached data
        draw_taskbar(frame, taskbar_cache['current_task'], taskbar_cache['next_task'], taskbar_cache)
        
        # Resize frame to make window larger
        height, width = frame.shape[:2]
        scale_factor = 1.5  # Make it 1.5x larger
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        cv2.imshow('Face Recognition', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()