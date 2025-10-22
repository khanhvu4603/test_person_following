import tf2onnx, tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
m = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
spec = (tf.TensorSpec((None,224,224,3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path="mb2_gap.onnx")
