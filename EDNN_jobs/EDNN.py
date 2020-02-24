#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ksprague1/Lattice-RNN/blob/master/Potts_EDNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# ednn.py, from http://kylemills.net/blog/deep-learning/ednn-part-2-the-ising-model-of-magnetism/
import sys
import tensorflow as tf
# ednn.py
class EDNN_helper(object):
    """breaks an image up into tiles for a smaller network"""
    def __init__(self, L, f, c):
        assert f <= L//2, "Focus must be less than half the image size to use this implementation."
        assert (f + 2*c) <= L, "Total tile size (f+2c) is larger than input image."
        self.l = L
        self.f = f
        self.c = c
    
    def __roll(self, in_, num, axis):
        """author: Kyle Mills"""
        D = tf.transpose(in_, perm=[axis, 1-axis])  #if axis=1, transpose first
        D = tf.concat([D[num:, :], D[0:num, :]], axis=0)
        return tf.transpose(D, perm=[axis, 1-axis]) #if axis=1, transpose back
    def __slice(self, in_, x1, y1, w, h): 
        """author: Kyle Mills"""
        return in_[x1:x1+w, y1:y1+h]
    def ednn_split(self, in_): 
        """author: Kyle Mills"""
        l = self.l
        f = self.f
        c = self.c
        tiles = []
        for iTile in range(l//f): #l/f tiles in each direction
            for jTile in range(l//f):
                #calculate the indices of the centre of
                #this tile (i.e. the centre of the focus region)
                cot = (iTile*f + f//2, jTile*f + f//2) #centre of tile
                foc_centered = in_ 
                #shift the picture, wrapping the image around,
                #so that the focus is centered in the middle of the image
                foc_centered = self.__roll(foc_centered, l//2-cot[0], 0)
                foc_centered = self.__roll(foc_centered, l//2-cot[1], 1)
                #Finally slice away the excess image that we don't want to appear in this tile
                final = self.__slice(foc_centered, l//2-f//2-c, l//2-f//2-c, 2*c+f, 2*c+f)
                tiles.append(final)
        return tf.expand_dims(tiles, axis=3)


# In[2]:


#First, some imports
import numpy as np
#import matplotlib.pyplot as plt
#import progressbar
#get_ipython().run_line_magic('matplotlib', 'inline')
import h5py

# In[3]:



# In[4]:
MODEL = sys.argv[1]

N=L=8
f=2
c=1
Q=3 if MODEL=="POTTS1" else 2
T=4
div=240
print(sys.argv,Q)
FOLDER = "/home/ksprague/Notebook/models/EDNN%s"%MODEL
dir_ ="/home/ksprague/Notebook/data/traindata/drop/%s"%MODEL
#for saving the ednn
SAVEDIR = '/home/ksprague/Notebook/models/EDNN%s/model.ckpt'%MODEL


# In[5]:


FileName = dir_+"EDNN-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,div,T)
def get_data(N,file = FileName):
    #assert N<=1600000# N must be less than 25000
    f = h5py.File(file, "r")
    data = f["Inputs"][:N]
    label = f["Labels"][:N]
    #data = F['data'][:N, ...,0]*1.0
    #label = F['energy'][:N, ...]*1.0
    return data, [[a] for a in label]


# In[6]:

# In[7]:


def NN(_in):
    """the network to be used on each tile of an image"""
    tile_size = f + 2*c
    _in = tf.reshape(_in, (-1, tile_size**2))
    nn = tf.contrib.layers.fully_connected(_in, 32)
    nn = tf.contrib.layers.fully_connected(nn, 64)    
    nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None)
    return nn


# In[8]:

#MAKING THE EDNN
#tf.reset_default_graph()
#data comes in a [ batch * L * L ] tensor, and labels a [ batch * 1] tensor
x = tf.placeholder(tf.float32, (None, L, L), name='input_image')
y = tf.placeholder(tf.float32, (None, 1))
helper = EDNN_helper(L=L, f=f, c=c)
#Then the EDNN-specific code:
tiles = tf.map_fn(helper.ednn_split, x, back_prop=False)
tiles = tf.transpose(tiles, perm=[1,0,2,3,4])
output = tf.map_fn(NN, tiles, back_prop=True)
output = tf.transpose(output, perm=[1,0,2])
predicted = tf.reduce_sum(output, axis=1)
#define the loss function
loss = tf.reduce_mean(tf.square(y-predicted))
#create an optimizer, a training op, and an init op
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()


# In[9]:


#grabbing data
data, labels = get_data(div*8000)
train_data,train_labels = data,labels
print(len(data))


# In[10]:


sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()
if sys.argv[-1] == "-l":
    saver.restore(sess,SAVEDIR)
    args = [int(x) for x in sys.argv[2:-1]]
else:
    args = [int(x) for x in sys.argv[2:]]
train_writer = tf.summary.FileWriter(FOLDER+'/train',sess.graph)


# In[11]:


BATCH_SIZE = 4000
losses = []
losses2 = []

def train(EPOCHS = (0,100)):
   """Trains the ednn"""
   global losses, losses2
   print("Starting")
   for epoch in range(*EPOCHS):
       for batch in range(train_data.shape[0] // BATCH_SIZE):
           _, loss_val = sess.run([train_step, loss],
                      feed_dict={
                           x: train_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE],
                           y: train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]})
           losses+=[loss_val]
           print(".",end="")
           #tf.summary.scalar('loss',loss_val)
           if batch%5==0:
               loss_val2 = test_loss()
               #tf.summary.scalar('test_loss',loss_val)
               print(loss_val2)
               sys.stdout.flush()
               saver.save(sess, save_path=SAVEDIR)
               summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss_val2),
                                          tf.Summary.Value(tag="train_loss", simple_value=loss_val)])
           else:
               summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=loss_val)])
           train_writer.add_summary(summary, (train_data.shape[0] // BATCH_SIZE)*epoch+batch)
       loss_val2 = [test_loss()]
       losses2+=loss_val2
       print(epoch, float(loss_val2[0]),end = "|")
   saver.save(sess, save_path=SAVEDIR)


# In[12]:


def Hs(arrs):
   """Runs the ednn on a set of states"""
   if Q!=2:
       arrs = [[[2*x/(Q-1) - 1 for x in y]for y in arr]for arr in arrs]
   pred = sess.run([predicted],feed_dict={x: arrs})
   return [a[0] for a in pred[0]]
def domain(n,q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be formatted for the rnn but not the hamiltonians"""
    if MODEL =="POTTS1":
        return [[np.random.randint(q) for a in range(n)]for b in range(n)]#potts
    else:
        return [[np.random.randint(0,2)*2-1 for a in range(n)]for b in range(n)]#ising

def H(arr):
    """Calculates a hamiltonian for ising (1st or 2nd nearest) or potts models"""
    #shift one to the right elements
    x = np.roll(arr,1,axis=1)
    #shift elements one down
    y = np.roll(arr,1,axis=0)
    #multiply original with transformations and sum each arr
    if MODEL =="ISING1":
        x = np.sum(np.multiply(arr,x))#Ising
        y = np.sum(np.multiply(arr,y))
    elif MODEL == "POTTS1":
        x = np.sum(np.cos(np.subtract(arr,x)*2*np.pi/Q))#Potts
        y = np.sum(np.cos(np.subtract(arr,y)*2*np.pi/Q))
    return -float(x+y)
def dH(Hs,arrs1,arrs2):#these are in original format
    """Calculates actual and RNN probability based Energy differences for two sets of equal sized arrays"""
    dH = np.asarray(Hs(arrs2))-np.asarray(Hs(arrs1))
    real_dH = np.asarray([H(arr) for arr in arrs2])-np.asarray([H(arr) for arr in arrs1])
    return dH,real_dH
def rmse(Hs,seed = 55,plot =False):
    """Graphs actual Energy differences Vs RNN generated energy differences"""
    np.random.seed(seed)
    ds = []
    ds = dH(Hs,[domain(N,Q)for n in range(1000)],[domain(N,Q)for n in range(1000)])
    mse = np.sum((ds[0]-ds[1])**2)/len(ds[0])
    rmse = mse**0.5
    print ('RMSE:', rmse)
    if plot:
        plt.plot(ds[1],ds[0],'g.')
        plt.plot(ds[1],ds[1],'r-')
        plt.xlabel("Expected ΔE")
        plt.ylabel("ΔE with RNN")
        plt.show()
    return rmse
def test_loss(seed=301):
    """my fast test loss function for a good idea at actual loss"""
    return rmse(Hs,seed,False)
    #np.random.seed(seed)
    #data = [domain(L,Q) for a in range(2000)]
    #inputs = [format_d(d) for d in data]
    #pred = sess.run([predicted],feed_dict={x: inputs})[0]
    #print(pred)
    #return np.sum([abs(pred[i][0]+real_H(data[i+1000])-pred[i+1000][0] - real_H(data[i])) for i in range(1000)])/1000


# In[ ]:

train(args)
print(test_loss())
print("done")
#print(losses)


