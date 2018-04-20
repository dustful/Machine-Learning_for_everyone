import tensorflow as tf

say = tf.constant("Hello, Tensorflow!")

sess = tf.Session()

print(sess.run(say))