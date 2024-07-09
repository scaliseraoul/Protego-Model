import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
    
class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)
  
class YamNetEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, yamnet_model_handle, **kwargs):
        super(YamNetEmbeddingLayer, self).__init__(**kwargs)
        self.yamnet_model = hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet')

    def call(self, inputs):
        _, embeddings, _ = self.yamnet_model(inputs)
        return embeddings

# Load the model
model = tf.keras.models.load_model('keras/train-neutral-4-trimmed-0-yammnet2.keras',custom_objects={'YamNetEmbeddingLayer': YamNetEmbeddingLayer,'ReduceMeanLayer': ReduceMeanLayer})

tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True)
# Get the input shape
input_shape = model.inputs[0].shape
print(input_shape)
input_shape = list(input_shape)
if input_shape[0] is None:
    input_shape[0] = 1

# Print model summary and input shape for debugging
print("Model Summary:")
model.summary()
print(f"Input shape: {input_shape}")

# Get the concrete function
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype))

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('sound_classifier_yamnet.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted successfully!")