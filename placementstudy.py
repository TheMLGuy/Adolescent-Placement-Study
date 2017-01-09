# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:47:57 2017

@author: Ashwin
"""
import pandas as pd
import tensorflow as tf
import numpy as np

#read data
placementDF=pd.read_excel('aps.xls')

#understand contents
placementDF.keys()
placementDF.head(2)


label=placementDF['PLACE']
placementDF=placementDF[['AGE','RACE','GENDER','NEURO','EMOT','DANGER','ELOPE','LOS','BEHAV','CUSTD','VIOL']]
#normalize age and behav

def normalize(col):
    col=(col-col.mean())/col.std()
    return col
#convert output label to one hot
originalY=label.unique()
oneHotLabel=[]
for j in label:
    labelSet=[]
    for i in originalY:
        if j==i:
            labelSet.append(1)
        else:
            labelSet.append(0)
    oneHotLabel.append(labelSet)        

    
placementDF['AGE']=normalize(placementDF['AGE'])
placementDF['BEHAV']=normalize(placementDF['BEHAV'])
placementDF['LOS']=normalize(placementDF['LOS'])
#train and test data
trainData=np.array(placementDF[0:355])
trainLabel=np.array(oneHotLabel[0:355])

testData=np.array(placementDF[355:508])
testLabel=np.array(oneHotLabel[355:508])

   
#yperparameters
learning_rate=0.01
epochs=500

#placeholders
X=tf.placeholder(tf.float32,[None,11])
y=tf.placeholder(tf.float32,[None,4])

#model weights and bias
W=tf.Variable(tf.zeros([11,4]))
b=tf.Variable(tf.zeros([4]))

out=tf.nn.softmax(tf.matmul(X,W)+b)

costFunction=tf.reduce_mean(-y*tf.log(out),reduction_indices=1)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(costFunction)

init=tf.global_variables_initializer()

#launch computation graph
with tf.Session() as ses:
    ses.run(init)
    for ep in range(epochs):
        avg_loss=0
        for batch_element in range(len(trainData)):
            _,c=ses.run([optimizer,costFunction],feed_dict={X:[trainData[batch_element]],y:[trainLabel[batch_element]]})
            avg_loss+=c
        print("Epoch "+str(ep)+" Loss "+str(avg_loss/len(trainData)))  
        
        
    prediction=tf.equal(tf.argmax(out,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(prediction,tf.float32))
    print("Accuracy in argmax "+str(accuracy.eval({X:testData,y:testLabel})*100))
    
#    correct_pred2 = tf.nn.in_top_k(out, tf.cast(tf.argmax(y,1), "int32"), 5)
#    accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))
#    print ("Accuracy of 'in top k' evaluation method " + str(accuracy2.eval({X:testData, y:testLabel})*100))
#    