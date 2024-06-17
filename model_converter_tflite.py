import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the model path
model_path = 'keras/train-neutral-4-trimmed-0-3dshape.keras'
tflite_model_path = 'keras/train-neutral-4-trimmed-0-3dshape.tflite'
# Load your Keras model
model = load_model(model_path)

# Define the input shape and create a concrete function
input_shape = (1, 345, 40)  # Update this based on your input shape
concrete_func = tf.function(lambda x: model(x))
concrete_func = concrete_func.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the converted model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_model_path}")

