import tensorflow as tf

x = tf.constant([[1.0, 2, 3], [3.0, 2, 1]])
tf.reduce_sum(x, 0)  # [2, 2, 2] ## [3, 3] by row, [1+1+1,1+1+1]
tf.reduce_sum(x, 1)  # [3, 3] ##[2, 2, 2] by colunm [1+1,1+1,1+1]
tf.reduce_sum(x, [0, 1])  # 6 ## all together 1+1 1+1 1+1 = 6, idea by column then row, or by row then column

#cosine
a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
b = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_b")
normalize_a = tf.nn.l2_normalize(a,0)        
normalize_b = tf.nn.l2_normalize(b,0)
cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b),0)
multiply=tf.multiply(a,b)
sess=tf.Session()
cos_sim=sess.run([cos_similarity,multiply],feed_dict={a:[1.0, 2, 3],b:[3.0, 2, 1]})

#lastnest api
import tensorflow as tf
import numpy as np


x = tf.constant(np.random.uniform(-1, 1, 10)) 
y = tf.constant(np.random.uniform(-1, 1, 10))
s = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)
print(tf.Session().run(s))



#norm
input_data = tf.constant([[1.0,2,3],[4.0,5,6],[7.0,8,9]])

output = tf.nn.l2_normalize(input_data, dim = 0) #dim 0 by column , dim 1 by row
sess=tf.Session()
print(sess.run(input_data))
print(sess.run(output))


#tf multiply the norm and then reducen_sum get the cosine
normalized_q_h_pool=tf.constant([[1.0,2,3],[4.0,5,6],[7.0,8,9]])
normalized_pos_h_pool=tf.constant([[3.0,2,1],[6.0,5,4],[9.0,8,7]])
norm_q = tf.nn.l2_normalize(normalized_q_h_pool,dim=1)
norm_p = tf.nn.l2_normalize(normalized_pos_h_pool,dim=1)
reduce = tf.reduce_sum(tf.multiply(norm_q,norm_p),1)
sess.run(norm_q) #by row [ 10,  73, 190]

#