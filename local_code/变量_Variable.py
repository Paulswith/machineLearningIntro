# -*- coding:utf-8 -*-
__author = ''
#__classOf 变量Variable

import tensorflow as tf


original_var = tf.Variable(0,name='counter')
#定义变量, 初始值-name
each_change_constant = tf.constant(23)
#定义一个常量,23

new_var = tf.add(original_var,each_change_constant)
#令变量和常量 加
updata_var = tf.assign(original_var,new_var)
#变量赋值

init = tf.initialize_all_variables()
#常规初始化变, 只有前面有定义变量都必须执行此步骤

with tf.Session() as session:
    session.run(init)
    #必须预执行一步, 来激活变量
    for step in xrange(3):
        session.run(updata_var) #执行的步骤
        print session.run(original_var) #每次输出都需要run

 