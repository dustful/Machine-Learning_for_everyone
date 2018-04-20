import tensorflow as tf

# placeholder : 레이지 기법으로, 프로그램이 실행되는 시점에 값을 정하는 함수
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = a + b # provides a shortcut for tf.add(a, b)

sess = tf.Session()

# feed_dict : 지칭할 매개변수명을 특정하는 인자
print(sess.run(add, feed_dict = {a:3, b:4.5}))
print(sess.run(add, feed_dict = {a:[1, 3], b:[2, 4]}))