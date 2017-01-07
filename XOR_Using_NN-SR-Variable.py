
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

xy = np.loadtxt('./test.txt' , unpack=True)
print xy


# In[3]:

x_data =np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1] , (4,1))
print x_data
print np.shape(x_data)
print y_data
print np.shape(y_data)


# In[4]:

X = tf.placeholder(tf.float32 , shape = [4,2])
Y= tf.placeholder(tf.float32  , shape = [4,1])


W1 = tf.Variable(tf.random_uniform([2,2],-1.0 , 1.0) , name = "W1")
b1= tf.Variable(tf.zeros([2]), name="Bias1")

W2 = tf.Variable(tf.random_uniform([2,1],-1.0 , 1.0) , name = "W2")
b2= tf.Variable(tf.zeros([1]), name="Bias2")



#2개의 데이터가 있으니깐  1, len(x_data) 2개의 데이터를 건네주고 -1 부터 1까지 준다 
# In[5]:

#W1 = tf.get_variable("W1" , [2,2] , initializer = tf.random_uniform([2,2],-1.0 , 1.0))
with tf.variable_scope('layer2'  ) as scope:
    W1 = tf.get_variable("W1" , [2,2] , initializer = tf.contrib.layers.xavier_initializer())


# In[11]:

#W1 = tf.get_variable("W1" , [2,2] , initializer = tf.random_uniform([2,2],-1.0 , 1.0))
with tf.variable_scope('layer2' , reuse=True  ) as scope:
    scope.reuse_variables()
    W1 = tf.get_variable("W1" , [2,2] , initializer = tf.contrib.layers.xavier_initializer())


# In[6]:

L2 = tf.sigmoid(tf.matmul(X , W1) +b1)
hypothesis = tf.sigmoid(tf.matmul(L2 , W2) +b2)
#hypothesis = tf.matmul(L2 , W2) +b2)


# In[7]:

learningRate= tf.Variable(0.1 , name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(learningRate)
#cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis , Y))
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1.-hypothesis)) # sigmoid function ?
cost_str=tf.cast(cost, tf.float32 ) 

train = optimizer.minimize(cost)
        
correct_prediction = tf.equal(tf.floor(hypothesis+0.5) , Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float64"))
saver =tf.train.Saver()
global_step=tf.Variable(0 , name = 'global_step', trainable = False) # global_step은 저장하지 않는다.


# In[ ]:




# In[8]:

init = tf.initialize_all_variables()


# In[9]:

sess= tf.Session()
sess.run(init)


# In[10]:


save_path = '/home/ncc/notebook/save'
save_name="model.ckpt"
    


# In[11]:

def is_restore(save_dir , save_name ):
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print "save data is exist! "
        print 'path :' ,(ckpt.model_checkpoint_path)
        
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    else:
        print "save data isn't exist! "
    
        


# In[12]:

#save_path = saver.save(sess, save_path) <-- 여기에다 이 코드를 삽입하면 안된다. sess.run 이후에 삽입해야 한다 


# In[16]:

print y_data
is_restore(save_path , save_name)


# In[20]:




# In[17]:


print y_data
#global_step=tf.Variable(0 , name = 'global_step', trainable = False) # global_step은 저장하지 않는다.

start =sess.run(global_step) 
print ('start from here{0}'.format(start))
for step in xrange(500):
    global_step.assign(step).eval(session=sess)
    sess.run(train, feed_dict={X:x_data , Y: y_data})
    
    saver.save(sess, save_path+save_name , global_step = global_step)
    #correct_prediction = tf.equal(tf.floor(hypothesis+0.5) , Y)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float"))
    #매 스탭마다 저장을 하면 이렇게 느려진다. 특정 조건에서만 저장하도록해야 한다 .
    if step %200 ==0:
    
        acc ,cost1=sess.run([accuracy,cost_str], feed_dict={X : x_data ,Y: y_data})
        print step ,acc   ,cost1       
        
# accuracy.eval({X:x_data , Y:y_data})



# In[ ]:




# In[ ]:

saver.restore(sess, save_path + save_name )
for i in range(10000):
    sess.run(train, feed_dict = {X:x_data ,Y:y_data})
    


# In[ ]:




# In[ ]:



