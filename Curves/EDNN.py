from EDNNHelper import EDNN_helper
import tensorflow as tf
"""THIS FILE IS FOR RUNNING EDNN MODELS, ALL YOU NEED IS EDNN.Hs(arrs)"""
# size of ising model, focus?, context
f=2
c=1
T=4
def NN(_in):
   """The network to be run on each tile"""
   tile_size = f + 2*c
   _in = tf.reshape(_in, (-1, tile_size**2))
   nn = tf.contrib.layers.fully_connected(_in, 32)
   nn = tf.contrib.layers.fully_connected(nn, 64)
   nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None)
   return nn
class EDNN():
   def __init__(self,SAVEDIR,N=8,Q=2):
       """Initialize the ednn"""
       L=N
       self.Q=Q
       self.x = tf.placeholder(tf.float32, (None, L, L), name='input_image')
       helper = EDNN_helper(L=L, f=f, c=c)
       #Then the EDNN-specific code:
       tiles = tf.map_fn(helper.ednn_split, self.x, back_prop=False)
       tiles = tf.transpose(tiles, perm=[1,0,2,3,4])
       output = tf.map_fn(NN, tiles, back_prop=True)
       output = tf.transpose(output, perm=[1,0,2])
       self.predicted = tf.reduce_sum(output, axis=1)
       self.sess = tf.InteractiveSession()
       init = tf.global_variables_initializer()
       self.sess.run(init)
       saver = tf.train.Saver()
       saver.restore(self.sess,SAVEDIR+"/model.ckpt")
   def Hs(self,arrs):
       """Run the ednn on a set of configurations"""
       if self.Q!=2:
           arrs = [[[2*x/(self.Q-1) - 1 for x in y]for y in arr]for arr in arrs]
       pred = self.sess.run([self.predicted],feed_dict={self.x: arrs})
       return [a[0] for a in pred[0]]
   def H(self,arr):
       """Run the ednn on a single configuration"""
       if self.Q!=2:
           arr = [[2*x/(self.Q-1) - 1 for x in y]for y in arr]
       pred = self.sess.run([self.predicted],feed_dict={self.x: [arr]})
       return pred[0][0][0]
   def close(self):
       """reset tf variables"""
       tf.reset_default_graph()
