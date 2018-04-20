import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_sum(tf.square(hypothesis - y))

learing_rate = 0.1
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - learing_rate * gradient
update = w.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_val = []
loss_val = []

for step in range(21):
    sess.run(update, feed_dict = {x:x_data, y:y_data})
    print(step, sess.run(loss, feed_dict = {x:x_data, y:y_data}), sess.run(w))

# results
# ====================
# 0 9.819557 [0.16250557]
# 1 2.793119 [0.55333626]
# 2 0.7944869 [0.76177937]
# 3 0.22598746 [0.872949]
# 4 0.064280875 [0.9322395]
# 5 0.01828432 [0.96386105]
# 6 0.0052008675 [0.9807259]
# 7 0.0014793568 [0.98972046]
# 8 0.00042079936 [0.99451756]
# 9 0.00011969403 [0.99707603]
# 10 3.4045224e-05 [0.99844056]
# 11 9.685284e-06 [0.9991683]
# 12 2.7549745e-06 [0.9995564]
# 13 7.836053e-07 [0.9997634]
# 14 2.2295534e-07 [0.9998738]
# 15 6.342215e-08 [0.9999327]
# 16 1.8050926e-08 [0.9999641]
# 17 5.1318985e-09 [0.99998087]
# 18 1.4507471e-09 [0.9999898]
# 19 4.0994408e-10 [0.9999946]
# 20 1.2046897e-10 [0.9999971]