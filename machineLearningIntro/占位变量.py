# -*- coding:utf-8 -*-
__author = ''

# __classOf 占位变量 placeholder


import tensorflow as tf

input1 = tf.placeholder(tf.float32)
# 占位一个值,在执行的时候传入, 这意味着与下面执行的时候传入的feed_dict, 这个字典基本是成对出现的
input2 = tf.placeholder(tf.float32)
# 可以指定数值类型


output = tf.matmul(input1, input2)
# tf.mul 乘法

with tf.Session() as session:
    print session.run(output, feed_dict={input1: [[10.0]], input2: [[22.3]]})
    # 在执行的时候赋值
    # 注意矩阵的写法, [[]]
