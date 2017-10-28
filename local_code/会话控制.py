# -*- coding:utf-8 -*-
__author = ''
#__classOf 会话控制

import tensorflow as tf


matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
#矩阵乘法, [3,3] * [[2],[2]] == [[12]]
product = tf.matmul(matrix1,matrix2)
#tf中的矩阵乘法 类似np.dot(matrix1,matrix2)

#method 1 常规的打开后关闭
# session = tf.Session()
# result = session.run(product)
# print result
# session.close()

#method 2 with open
with tf.Session() as tf:
    result = tf.run(product)
    print result

 