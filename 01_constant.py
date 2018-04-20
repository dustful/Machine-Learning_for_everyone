import tensorflow as tf

# build graph using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)

# feed data and run graph(operation)
sess = tf.Session()

# update variables in the graph(and return values)
print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run(node3))