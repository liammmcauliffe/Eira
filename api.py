#!/usr/bin/env python3
"""
Simple REST API for Eira Face Recognition System
Provides CRUD operations for all database tables
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import os
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Supabase client
def init_supabase() -> Client:
    """Initialize Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    return create_client(supabase_url, supabase_key)

supabase = init_supabase()

@app.route('/api/persons', methods=['GET'])
def get_persons():
    """Get all persons"""
    try:
        result = supabase.table("persons").select("*").execute()
        return jsonify({
            "success": True,
            "data": result.data,
            "count": len(result.data)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/persons/<int:person_id>', methods=['GET'])
def get_person(person_id):
    """Get a specific person by ID"""
    try:
        result = supabase.table("persons").select("*").eq("id", person_id).execute()
        if result.data:
            return jsonify({"success": True, "data": result.data[0]})
        else:
            return jsonify({"success": False, "error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/persons', methods=['POST'])
def create_person():
    """Create a new person"""
    try:
        data = request.get_json()
        required_fields = ['name', 'relationship', 'image_filename']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        result = supabase.table("persons").insert(data).execute()
        return jsonify({
            "success": True,
            "data": result.data[0],
            "message": "Person created successfully"
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/persons/<int:person_id>', methods=['PUT'])
def update_person(person_id):
    """Update a person"""
    try:
        data = request.get_json()
        result = supabase.table("persons").update(data).eq("id", person_id).execute()
        
        if result.data:
            return jsonify({
                "success": True,
                "data": result.data[0],
                "message": "Person updated successfully"
            })
        else:
            return jsonify({"success": False, "error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/persons/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person and their photo"""
    try:
        # First get the person to find their image filename
        person_result = supabase.table("persons").select("image_filename").eq("id", person_id).execute()
        
        if not person_result.data:
            return jsonify({"success": False, "error": "Person not found"}), 404
        
        image_filename = person_result.data[0]['image_filename']
        
        # Delete the person from database
        result = supabase.table("persons").delete().eq("id", person_id).execute()
        
        if result.data:
            # Try to delete the photo from storage (don't fail if photo doesn't exist)
            try:
                supabase.storage.from_("face-photos").remove([image_filename])
            except Exception as photo_error:
                print(f"Warning: Could not delete photo {image_filename}: {photo_error}")
            
            return jsonify({
                "success": True, 
                "message": "Person and photo deleted successfully",
                "deleted_filename": image_filename
            })
        else:
            return jsonify({"success": False, "error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks"""
    try:
        result = supabase.table("tasks").select("*").order("scheduled_time").execute()
        return jsonify({
            "success": True,
            "data": result.data,
            "count": len(result.data)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """Get a specific task by ID"""
    try:
        result = supabase.table("tasks").select("*").eq("id", task_id).execute()
        if result.data:
            return jsonify({"success": True, "data": result.data[0]})
        else:
            return jsonify({"success": False, "error": "Task not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks', methods=['POST'])
def create_task():
    """Create a new task"""
    try:
        data = request.get_json()
        required_fields = ['title', 'scheduled_time']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        result = supabase.table("tasks").insert(data).execute()
        return jsonify({
            "success": True,
            "data": result.data[0],
            "message": "Task created successfully"
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """Update a task"""
    try:
        data = request.get_json()
        result = supabase.table("tasks").update(data).eq("id", task_id).execute()
        
        if result.data:
            return jsonify({
                "success": True,
                "data": result.data[0],
                "message": "Task updated successfully"
            })
        else:
            return jsonify({"success": False, "error": "Task not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task"""
    try:
        result = supabase.table("tasks").delete().eq("id", task_id).execute()
        if result.data:
            return jsonify({"success": True, "message": "Task deleted successfully"})
        else:
            return jsonify({"success": False, "error": "Task not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get all face detections"""
    try:
        result = supabase.table("face_detections").select("*, persons(*)").order("detected_at", desc=True).execute()
        return jsonify({
            "success": True,
            "data": result.data,
            "count": len(result.data)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detections', methods=['POST'])
def create_detection():
    """Create a new face detection"""
    try:
        data = request.get_json()
        required_fields = ['person_id']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        result = supabase.table("face_detections").insert(data).execute()
        return jsonify({
            "success": True,
            "data": result.data[0],
            "message": "Detection logged successfully"
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detections/<int:person_id>/recent', methods=['GET'])
def get_recent_detections(person_id):
    """Get recent detections for a specific person"""
    try:
        result = supabase.table("face_detections").select("*").eq("person_id", person_id).order("detected_at", desc=True).limit(10).execute()
        return jsonify({
            "success": True,
            "data": result.data,
            "count": len(result.data)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/encodings/<int:person_id>', methods=['GET'])
def get_encoding(person_id):
    """Get face encoding for a specific person"""
    try:
        result = supabase.table("face_encodings").select("*").eq("person_id", person_id).execute()
        if result.data:
            return jsonify({"success": True, "data": result.data[0]})
        else:
            return jsonify({"success": False, "error": "Encoding not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/encodings', methods=['POST'])
def create_encoding():
    """Create a new face encoding"""
    try:
        data = request.get_json()
        required_fields = ['person_id', 'encoding_data']
        
        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Delete old encoding if exists
        supabase.table("face_encodings").delete().eq("person_id", data['person_id']).execute()
        
        result = supabase.table("face_encodings").insert(data).execute()
        return jsonify({
            "success": True,
            "data": result.data[0],
            "message": "Encoding created successfully"
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload-photo', methods=['POST'])
def upload_photo():
    """Upload a photo to Supabase storage"""
    try:
        data = request.get_json()
        
        if 'image_data' not in data or 'filename' not in data:
            return jsonify({"success": False, "error": "Missing image_data or filename"}), 400
        
        # Decode base64 image data
        image_data = data['image_data']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Upload to Supabase storage
        result = supabase.storage.from_("face-photos").upload(
            data['filename'],
            image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        if result:
            return jsonify({
                "success": True,
                "message": "Photo uploaded successfully",
                "filename": data['filename']
            })
        else:
            return jsonify({"success": False, "error": "Failed to upload photo"}), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/delete-photo/<filename>', methods=['DELETE'])
def delete_photo(filename):
    """Delete a photo from Supabase storage"""
    try:
        # Delete from Supabase storage
        result = supabase.storage.from_("face-photos").remove([filename])
        
        if result:
            return jsonify({
                "success": True,
                "message": "Photo deleted successfully",
                "filename": filename
            })
        else:
            return jsonify({"success": False, "error": "Failed to delete photo"}), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        supabase.table("persons").select("id").limit(1).execute()
        return jsonify({
            "success": True,
            "message": "API is healthy",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        persons_count = len(supabase.table("persons").select("id").execute().data)
        tasks_count = len(supabase.table("tasks").select("id").execute().data)
        detections_count = len(supabase.table("face_detections").select("id").execute().data)
        encodings_count = len(supabase.table("face_encodings").select("id").execute().data)
        
        return jsonify({
            "success": True,
            "data": {
                "persons": persons_count,
                "tasks": tasks_count,
                "detections": detections_count,
                "encodings": encodings_count
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"success": False, "error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Eira API Server...")
    print("Available endpoints:")
    print("   GET    /api/persons              - Get all persons")
    print("   GET    /api/persons/<id>         - Get specific person")
    print("   POST   /api/persons              - Create person")
    print("   PUT    /api/persons/<id>         - Update person")
    print("   DELETE /api/persons/<id>         - Delete person")
    print("   GET    /api/tasks                - Get all tasks")
    print("   GET    /api/tasks/<id>           - Get specific task")
    print("   POST   /api/tasks                - Create task")
    print("   PUT    /api/tasks/<id>           - Update task")
    print("   DELETE /api/tasks/<id>           - Delete task")
    print("   GET    /api/detections           - Get all detections")
    print("   POST   /api/detections           - Log detection")
    print("   GET    /api/detections/<id>/recent - Get recent detections")
    print("   GET    /api/encodings/<id>       - Get face encoding")
    print("   POST   /api/encodings            - Create face encoding")
    print("   GET    /api/health               - Health check")
    print("   GET    /api/stats                - Database statistics")
    print("")
    print("Server will be available at: http://localhost:5000")
    print("API Documentation: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
