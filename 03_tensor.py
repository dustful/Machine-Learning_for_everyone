import tensorflow as tf

# tensor ranks, (shapes), types
# ====================
# rank 0, scalar
# ex)
s = 12345

# rank 1, vector
# ex)
v = [678, 910]

# rank 2, matrix
# ex)
m = [[12, 34], [56, 78]]

# rank 3, 3-tensor
# ex)
t = [[[1], [2], [3]], [[4], [5], [6]]]

# rank 4, n-tensor
# ...
# ====================
# 32 bits floating point
tf.float32

# 64 bits floating point
tf.float64

# 8 bits signed integer
tf.int8

# 16 bits signed integer
tf.int16

# 32 bits signed integer
tf.int32

# 64 bits signed integer
tf.int64