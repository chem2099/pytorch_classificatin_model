import tensorflow as tf
t1=tf.constant([[1,1,1],[2,2,2]],dtype=tf.int32)
t2=tf.constant([[3,3],[4,4]],dtype=tf.int32)
#t3=t1+t2
#t4=tf.concat([t1, t2], 0)
t5=tf.concat([t1, t2], 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for temp in [t1,t2,t5]:
        print ('\n',sess.run(temp))
