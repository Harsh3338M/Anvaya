import os
import json
import numpy as np
import joblib

def normalize_raw_landmarks(raw_landmarks):
    """
    Normalizes a flat 63-element list/array of raw keypoints (x0, y0, z0, ...):
    1. Origin shifts to the wrist (landmark 0).
    2. Scaling to make it scale-invariant by dividing by the max distance from wrist.
    """
    # Reshape flat list of 63 coordinates into (21, 3)
    coords = np.array(raw_landmarks).reshape(21, 3)
    
    # 1. Origin shift (wrist is coords[0])
    wrist = coords[0]
    shifted_coords = coords - wrist
    
    # 2. Scale Invariance (divide by max distance from wrist)
    distances = np.linalg.norm(shifted_coords, axis=1)
    max_dist = np.max(distances)
    if max_dist == 0:
        max_dist = 1.0
        
    normalized_coords = shifted_coords / max_dist
    return normalized_coords.flatten()

def load_gesture_classifier():
    """
    Search for trained model files and classes mapping in local paths.
    Returns: (model, classes_list, model_type)
    """
    # Search paths for gesture_model.pkl / classes.json
    possible_paths = [
        # Current directory
        (".", "gesture_model.pkl", "classes.json"),
        # If run from root directory
        ("model_training", "gesture_model.pkl", "classes.json"),
        # Relative to backend in typical project layout
        ("../model_training", "gesture_model.pkl", "classes.json"),
    ]
    
    model = None
    classes_list = None
    model_type = None
    
    # 1. Attempt to load Scikit-Learn MLP
    for base_dir, model_name, classes_name in possible_paths:
        m_path = os.path.join(base_dir, model_name)
        c_path = os.path.join(base_dir, classes_name)
        
        if os.path.exists(m_path) and os.path.exists(c_path):
            try:
                model = joblib.load(m_path)
                with open(c_path, 'r') as f:
                    classes_list = json.load(f)
                model_type = "sklearn"
                print(f"Successfully loaded Scikit-Learn model from {m_path}")
                return model, classes_list, model_type
            except Exception as e:
                print(f"Error loading Sklearn model from {m_path}: {e}")
                
    # 2. Attempt to load Keras MLP (optional fallback)
    possible_keras_paths = [
        (".", "gesture_model.h5", "classes.json"),
        ("model_training", "gesture_model.h5", "classes.json"),
        ("../model_training", "gesture_model.h5", "classes.json"),
    ]
    
    for base_dir, model_name, classes_name in possible_keras_paths:
        m_path = os.path.join(base_dir, model_name)
        c_path = os.path.join(base_dir, classes_name)
        
        if os.path.exists(m_path) and os.path.exists(c_path):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(m_path)
                with open(c_path, 'r') as f:
                    classes_list = json.load(f)
                model_type = "keras"
                print(f"Successfully loaded Keras model from {m_path}")
                return model, classes_list, model_type
            except Exception as e:
                print(f"Error loading Keras model from {m_path}: {e}")
                
    return None, None, None
