"""
Resave the model with current TensorFlow version for compatibility.
Run this notebook/script with the TF version you want to deploy with.
"""
import tensorflow as tf

# Load old model
model = tf.keras.models.load_model('model.h5')

# Save in new native format (.keras)
model.save('model.keras')

# Or save in SavedModel format (directory)
# model.save('saved_model/')

print(f"Model re-saved successfully with TensorFlow {tf.__version__}")
