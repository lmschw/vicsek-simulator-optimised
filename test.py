import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4)
a1 = [[1, 1], [1.1, 1.1], [1, 0]]
a2 = [[2, 2], [2.2, 2.2], [0, 1]]

t = tf.constant([a1, a2])

print("actual print:")
print(t)