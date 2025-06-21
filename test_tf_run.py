import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a constant tensor
hello = tf.constant("Hello, TensorFlow!")
print(hello.numpy().decode())