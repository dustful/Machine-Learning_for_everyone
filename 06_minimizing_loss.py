import tensorflow as tf
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

w = tf.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_val = []
loss_val = []

for i in range(-30, 50):
    feed_w = i * 0.1
    curr_loss, curr_w = sess.run([loss, w], feed_dict = {w:feed_w})
    w_val.append(curr_w)
    loss_val.append(curr_loss)

plt.plot(w_val, loss_val)
plt.show()