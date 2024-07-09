import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

class YamNetEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, yamnet_model_handle, **kwargs):
        super(YamNetEmbeddingLayer, self).__init__(**kwargs)
        self.yamnet_model = hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet')

    def call(self, inputs):
        _, embeddings, _ = self.yamnet_model(inputs)
        reshaped_embeddings = tf.expand_dims(embeddings, axis=1)
        return reshaped_embeddings

# Define the model path
model_path = 'keras/train-neutral-4-trimmed-0-yamnet-combined.keras'
tflite_model_path = 'keras/train-neutral-4-trimmed-0-yamnet-combined.tflites'
# Load your Keras model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
model = load_model(model_path, custom_objects={'YamNetEmbeddingLayer': YamNetEmbeddingLayer(yamnet_model_handle)})

# Define the input shape and create a concrete function
input_shape = (None,)  # Update this based on your input shape
concrete_func = tf.function(lambda x: model(x))
concrete_func = concrete_func.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the converted model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_model_path}")

