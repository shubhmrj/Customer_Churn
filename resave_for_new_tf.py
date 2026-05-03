"""
Resave model with TensorFlow 2.20.0 for Python 3.13 compatibility
Run this after installing new TF version
"""
import tensorflow as tf
import os

print(f"TensorFlow version: {tf.__version__}")

# Load old model (H5 format is more portable across versions)
old_model_path = 'Models/model.h5'
new_model_path = 'Models/model.keras'

try:
    model = tf.keras.models.load_model(old_model_path)
    print(f"✓ Loaded model from {old_model_path}")
    
    # Save in new native format
    model.save(new_model_path)
    print(f"✓ Saved model to {new_model_path}")
    print("Model ready for deployment!")
except Exception as e:
    print(f"Error: {e}")
    print("Try: pip install tensorflow==2.20.0 first")
