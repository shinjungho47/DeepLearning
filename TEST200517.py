#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf


Train=pd.read_csv("C:/Users/shinj/Desktop/Train_data.csv")
Target=pd.read_csv("C:/Users/shinj/Desktop/Target_data.csv")
Train_data= np.array(Train, dtype=np.float32)
Target_data= np.array(Target, dtype=np.float32)


#print(Train_data)
#print(Target_data)

tf.compat.v1.reset_default_graph()
learning_rate=0.01
training_cnt=10000

X=tf.compat.v1.placeholder(tf.float32,[None,6])
Y=tf.compat.v1.placeholder(tf.float32,[None,1])

W1=tf.Variable(tf.random.normal([6,24]),name='weight1')
b1=tf.Variable(tf.random.normal([24]),name='bias1')
L1=tf.nn.relu(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random.normal([24,24]),name='weight2')
b2=tf.Variable(tf.random.normal([24]),name='bias2')
L2=tf.nn.relu(tf.matmul(L1,W2)+b2)

W3=tf.Variable(tf.random.normal([24,24]),name='weight3')
b3=tf.Variable(tf.random.normal([24]),name='bias3')
L3=tf.nn.relu(tf.matmul(L2,W3)+b3)

W4=tf.Variable(tf.random.normal([24,24]),name='weight4')
b4=tf.Variable(tf.random.normal([24]),name='bias4')
L4=tf.nn.relu(tf.matmul(L3,W4)+b4)

W5=tf.Variable(tf.random.normal([24,1]),name='weight5')
pred = tf.nn.softmax(tf.matmul(L4,W5))

delta=0.0001


cost = -tf.reduce_sum(Y * tf.math.log(pred+delta))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate)
op_train =optimizer.minimize(cost)

#predicted=tf.cast(pred>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(pred,Y),dtype=tf.float32))

sess=tf.compat.v1.Session()
init=tf.initialize_all_variables()
sess.run(init)

for step in range(training_cnt):
    sess.run(op_train,feed_dict={X:Train_data,Y:Target_data})
    if step %1000==0:
        print(step,sess.run(cost,feed_dict={X:Train_data,Y:Target_data}),sess.run([W1,W2,W3,W4,W5]))

p,a=sess.run([pred,accuracy],feed_dict={X:Train_data,Y:Target_data})
print("\nPred:",p,"\nAccuracy:",a)
