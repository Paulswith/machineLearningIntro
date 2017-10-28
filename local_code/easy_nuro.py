# -*- coding:utf-8 -*-
__author = ''
#__classof 兴趣激活
import tensorflow as tf
import numpy as np



x_data = np.random.rand(100).astype(np.float32)
# print x_data #随机生成100个浮点32位的数
y_data = x_data *0.1 + 0.3
# print y_data  #预期是-> y = W*x+b ,Weight & biases 预期是接近0.1  和 0.3


#mark ----------------create tensorflow structure -----start

Weights_arg = tf.random_uniform([1,1],-1.0,1.0) # 默认位浮点类型
# print Weights_arg  #参数: 一维, 初始范围始于-1,终于1
Weights = tf.Variable(Weights_difine)
# print Weights #Variable生成该变量
biases_arg = tf.zeros([1,1])
# print biases_arg  #初始范围为0-一维
biases = tf.Variable(biases_arg)
# print biases
y = Weights * x_data + biases
# print y  #代入W*x+b 提取y

loss_arg = tf.square(y - y_data)
# print loss_arg  #提取到的y 减去预期的y_data
loss = tf.reduce_mean(loss_arg)
# print loss   #减少loss

optimizer = tf.train.GradientDescentOptimizer(0.5)
# print optimizer   #创建一个减少误差范围的优化器  0.5是学习效率,或者学习率
train = optimizer.minimize(loss)
# print train   #用这个优化器进行每一步的优化处理,逐渐减少每次的误差

init = tf.initialize_all_variables()
#初始化整个结构

#mark ----------------create tensorflow structure -----end



session = tf.Session() #创建一个session
session.run(init)  #激活


#mark ----------------------run--------------------

for step in xrange(1000):
    session.run(train)
    if step % 20 == 0:
        print '{step}-->Weight:{Weight}-->biases:{biases}'.format(step=step,\
                                                             Weight=session.run(Weights),\
                                                             biases=session.run(biases))

 