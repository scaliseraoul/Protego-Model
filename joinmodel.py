import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

class YamNetEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, yamnet_model_handle, **kwargs):
        super(YamNetEmbeddingLayer, self).__init__(**kwargs)
        self.yamnet_model = hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet')

    def call(self, inputs):
        _, embeddings, _ = self.yamnet_model(inputs)
        return embeddings

# Define paths
model_path = 'keras/train-neutral-4-trimmed-0-yammnet.keras'
saved_model_path = 'keras/train-neutral-4-trimmed-0-yamnet-combined.keras'
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'

# Load your custom model
model = load_model(model_path)

# Define input for the combined model
input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')

# Create YamNetEmbeddingLayer and extract embeddings
yamnet_embedding_layer = YamNetEmbeddingLayer(yamnet_model_handle)
embeddings_output = yamnet_embedding_layer(input_segment)
serving_outputs = model(embeddings_output)
# Create and save the combined model
serving_model = tf.keras.Model(inputs=input_segment, outputs=serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

# Verify that the model has been saved
tf.keras.utils.plot_model(serving_model, to_file='model_structure.png', show_shapes=True)
