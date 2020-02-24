#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
import numpy as np
import h5py
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


#tf.enable_eager_execution()
from tensorflow.python.client import device_lib



CODE, FLATTEN, MODEL = sys.argv[1:4]
N = 8  # NxN spin model
#8x8 transitions at about 2.6, 4x4 at 4.0
Q = 3 if MODEL=="POTTS1" else 2  # Potts model, number of angles
KB = 1.0
# this line can take values of: POTTS1 or ISING1 or ISING2
# ISING1 means ising nearest neighbour
# ISING2 means ising second nearest neighbour
# POTTS1 means potts model, 1 neighbour

#Converting command line arguments into a dictionary
kwargs = {}
for arg in sys.argv[4:]:
    name,val = arg.split("=")
    try:
        kwargs[name] = val
        kwargs[name] = float(val)
        kwargs[name] = int(val)
    except:
        pass
            
print(sys.argv)
# In[4]:
sys.stdout.flush()
class Config():
    __Ising={
    'learning_rate': 0.0005,
    'num_layers' : 1,
    'num_steps' : N**2 - 1,
    'hidden_size' : 256,
    'keep_prob' :1.0,#0.9,
    "batch_size" : 8000,
    "vocab_size" : 1,
    "rnn_mode":"gru",
    "DSIZE":128,
    "RUNS":1,
    "EPOCHS":50,
    "device":'/gpu:0',
    "seed":12,
    "T":4
    }
    __Potts={
    'learning_rate': 0.00025,
    'hidden_size' : 512,
    "batch_size" : 4000,
    "vocab_size" : Q,
    "DSIZE":256,
    }
    def __init__(self,ispotts,**kwargs):
        """Uses the parameters in kwargs as config. if an argument is not specified, the default for said model is used"""
        self.__dict__.update(Config.__Ising)
        if ispotts:
            self.__dict__.update(Config.__Potts)
        self.__dict__.update(kwargs)
        global T
        T=self.T

        
ConfigIsing = Config(False)
ConfigPotts = Config(True)
ConfigIsingBIG = Config(False,hidden_size = 378,DSIZE=256,RUNS=4)

CONFIG = Config(MODEL=="POTTS1",**kwargs)#ConfigIsingBIG if MODEL[:-1] in ["ISING","SPINS"] else ConfigPotts
print(T)
print(CONFIG.__dict__)
DSIZE = CONFIG.DSIZE # the size of the dataset to be generated in 100k
RUNS = CONFIG.RUNS
#from google.colab import drive
#drive.mount('/content/gdrive')

# make a directory in google drive to save it so that 
# we don't need to regenerate it all the time
#code above is only if running on colab


#These are all the directories used in this notebook

FOLDER = "/home/ksprague/Notebook/data/traindata/"+CODE
DIR = "/home/ksprague/Notebook/models/"+(CODE+MODEL)
dir_ ="/home/ksprague/Notebook/data/traindata/%s"%(CODE+MODEL)
#should probably change dir when switching from ising to potts models
def file_name(div):
    if MODEL == "ISING1":
        return FOLDER+ "traindata-%dx%d-%d-%.1f.hdf5"%(N,N,div,T)
    elif MODEL == "ISING2":
        return FOLDER+"Ising2nd-%dx%d-%d-%.1f.hdf5"%(N,N,div,T)
    elif MODEL == "SPINS1":
        return FOLDER+"spinglass-%dx%d-%d-%.1f.hdf5"%(N,N,div,T)
    elif MODEL == "POTTS1":
        return FOLDER+"potts-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,div,T)
    elif MODEL == "SPINS2":
        return FOLDER+"spinglass2-%dx%d-%d-%.1f.hdf5"%(N,N,div,T)
# In[5]:


class SpinGlass():
  """Calculates the hamiltonian for a nn spinglass model"""
  def __init__(self,seed,N):
    """The random couplling energies are chosen according to the seed"""
    self.N = N
    np.random.seed(seed)
    self.h = np.random.normal(0,1,[N,N])
    self.v = np.random.normal(0,1,[N,N])
    #print(self.h,'\n',self.v)
    
  def H(self,arr):
      #shift one to the right elements
      x = np.roll(arr,1,axis=1)
      #shift elements one down
      y = np.roll(arr,1,axis=0)
      #multiply original with transformations and sum each arr
      x = np.sum(np.multiply(np.multiply(arr,x),self.h))#Ising
      y = np.sum(np.multiply(np.multiply(arr,y),self.v))
      return -float(x+y)
  def __str__(self):
      s = np.arange(self.N**2).reshape([self.N,self.N])+1
      sx = np.roll(s,1,axis=1)
      sy = np.roll(s,1,axis=0)
      s1 = "".join(["%d %d %f\n"%(s[a][b],sx[a][b],self.h[a][b]) for a in range(self.N) for b in range(self.N)])
      s2 = "".join(["%d %d %f\n"%(s[a][b],sy[a][b],self.v[a][b])for a in range(self.N) for b in range(self.N)])
      return s1+s2
class SpinGlass2():
    """Calculates the hamiltonian for an all interacting spinglass model"""
    def __init__(self,seed,N):
        """The random couplling energies are chosen according to the seed"""
        self.N = N**2
        np.random.seed(seed)
        self.c= np.random.normal(0,1,[self.N]*2)
        for i in range(self.N):
            for j in range(self.N):
                if j<=i:
                    self.c[i][j]=0
    def H(self,arr):
        a=np.reshape(arr,[self.N,1])
        b=np.reshape(arr,[1,self.N])
        return -np.sum(np.multiply(self.c,np.matmul(a,b)))
    def __str__(self):
        string=""
        for i in range(self.N):
            for j in range(self.N):
                if self.c[i][j]!=0:
                    string+= "%d %d %f\n"%(i+1,j+1,self.c[i][j])
        return string
G2 = SpinGlass2(CONFIG.seed,8)
if MODEL=="SPINS2":
    print(G2)
G = SpinGlass(CONFIG.seed,8)#default is 12
def H(arr):
    """Calculates a hamiltonian for ising (1st or 2nd nearest) or potts models"""
    if MODEL == "ISING2":
        return H2(arr)
    if MODEL=="SPINS1":
        return G.H(arr)
    if MODEL=="SPINS2":
        return G2.H(arr)
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
def H2(arr):
    """Calculates a 2nd-nearest-neighbor hamiltonian for the ising model"""
    #shift one to the right elements
    x = np.roll(arr,1,axis=1)
    #shift elements one down
    y = np.roll(arr,1,axis=0)
    #multiply original with transformations and sum each arr
    w = np.roll(x,1,axis=0)
    # w and z are diaganol up and diaganol down
    z = np.roll(x,-1,axis=0)
    x = np.sum(np.multiply(arr,x))
    y = np.sum(np.multiply(arr,y))
    z = np.sum(np.multiply(arr,z))
    w = np.sum(np.multiply(arr,w))
    return -float(x+y) -float(z+w)/2

#TODO: define what this function does or is for  
def snake(arr):
    """flattens a 2d array into a 1d in a snake pattern """
    snake_ = []
    n = 1
    for a in arr:
        snake_ += a[::n]
        n*=-1
    return snake_
def spiral(arr,growing=True):
    """flattens 2d array into a 1d via a spiral pattern"""
    i=j=len(arr)//2
    k=-1
    spiral_ = [arr[i][j]]
    for s in range(1,len(arr)):
        for sy in range(s):
            i+=k
            spiral_ += [arr[i][j]]
        for sx in range(s):
            j+=k
            spiral_ += [arr[i][j]]
        k*=-1
    for sx in range(len(arr)-1):
        i+=k
        spiral_ += [arr[i][j]]
    return spiral_
def flatten(arr):
    if FLATTEN=="SPIRAL":
        return spiral(arr)
    else:
        return snake(arr)
#TODO: define what this function does or is for
def conv_inp(arrs):
    """Readies the input for an rnn. The array is flattened with the first n*n-1 being inputs and last n*n-1 being labels"""
    if MODEL[:5] in ["ISING","SPINS"]:
        out = [[[n] for n in flatten(arr)] for arr in arrs]#Ising
    elif MODEL =="POTTS1":
        out = [flatten(arr) for arr in arrs]#Potts
    in_ = [arr[:-1] for arr in out] 
    out = [arr[1:] for arr in out] 
    return in_,out

def domain(n,q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be formatted for the rnn but not the hamiltonians"""
    if MODEL =="POTTS1":
        return [[np.random.randint(q) for a in range(n)]for b in range(n)]#potts
    else:
        return [[np.random.randint(0,2)*2-1 for a in range(n)]for b in range(n)]#ising
    
def to_list(arr):
    """makes each entry in the array a Q wide array (all zeroes
    except the index of the original value)
    Ex: [[1,2]] becomes [[[0,1,0],[0,0,1]]]"""
    return [[rndarr(Q,i) for i in a] for a in arr]

#TODO: what is this function for
def rndarr(Q,i):
    """Creates a zero array of size Q and sets index i to 1"""
    arr = [0]*Q
    arr[i] = 1
    return arr
print(G)
#a = np.asarray([[3,2,9],[4,1,8],[5,6,7]])
#print(a,'\n',flatten(a))


# In[6]:


#TODO: what is this function for?
# it looks like a spin flip
def augment(arr):
    """Spin flip where a random spin in the grid is changed (Potts is still an integer representation)"""
    x,y = np.shape(arr)
    arr = [x.copy() for x in arr]
    x_ = np.random.randint(0,x)
    y_ = np.random.randint(0,y)
    if MODEL =="POTTS1": 
      arr[x_][y_] += np.random.randint(1,Q)
      arr[x_][y_] %= Q
    else:
      arr[x_][y_]*= -1
    return arr
  

def metropolis(x,y,T,iterate = 2500,start = 0):
    """Runs a monte carlo simulation and yields every grid after 'start' iterations have occured"""
    C = np.e**(-1.0/(KB*T))
    arr = arr = domain(x,Q)
    Hi = H(arr)
    for i in range(iterate):
        arr2 = augment(arr)
        Hf = H(arr2)
        dH = Hf-Hi
        n=np.random.random()
        if dH <=0 or n < C**dH:
            Hi = Hf
            arr = arr2
        if i>= start:
            if MODEL == "POTTS1":
                yield to_list(arr)
            else:
                yield arr


# this function generates the training data. It takes an arguement which says
# how many groups of 100k samples should be done

def h5gen(div = 32,runs=1):
    """generate samples for an NxN ising model at a specific T, throw away some configurations"""
    assert runs*(div//runs)==div
    div_multiplier = 100000
    length = div_multiplier*div
    f = h5py.File(file_name(div), "w")
    for i in range(div):
        if i%(div//runs)==0:
            print("||",end="")
            arrs = metropolis(N,N,T,iterate = div_multiplier*div//runs+110000,start = 110000)
        batch = [next(arrs) for x in range(length//div)]
        if MODEL == "POTTS1":
          out = [flatten(arr) for arr in batch]
        else:
          out = [[[n] for n in flatten(arr)] for arr in batch]
        in_ = np.asarray([arr[:-1] for arr in out],dtype=np.int8)  
        out = np.asarray([arr[1:] for arr in out],dtype=np.int8)
        if i==0:
            if MODEL == "POTTS1":
              print(in_.shape)
              f.create_dataset("Inputs", data=in_, chunks=True,maxshape=(length, N**2-1,Q))
              f.create_dataset("Labels", data=out, chunks=True,maxshape=(length, N**2-1,Q))
            else:
              f.create_dataset("Inputs", data=in_, chunks=True,maxshape=(length, N**2-1,1))
              f.create_dataset("Labels", data=out, chunks=True,maxshape=(length, N**2-1,1))
            f.close()
            f = h5py.File(file_name(div), "a")
        else:
            f["Inputs"].resize((f["Inputs"].shape[0] + length // div), axis = 0)
            f["Inputs"][-length // div:] = in_
            f["Labels"].resize((f["Labels"].shape[0] + length // div), axis = 0)
            f["Labels"][-length // div:] = out
        print((i+1)/div*100, end = "% ")
        sys.stdout.flush()
    f.close()
    print('Generation of MC data is complete')
    f.close()
if RUNS>0:
    h5gen(div = DSIZE,runs=RUNS)


#opening training data

if MODEL == "SPINS1":
    f = h5py.File(FOLDER+"spinglass-%dx%d-%d-%.1f.hdf5"%(N,N,DSIZE,T), "r")
if MODEL == "SPINS2":
    f = h5py.File(FOLDER+"spinglass2-%dx%d-%d-%.1f.hdf5"%(N,N,DSIZE,T), "r")
elif CONFIG.vocab_size ==1:
    f = h5py.File(FOLDER+"traindata-%dx%d-%d-%.1f.hdf5"%(N,N,DSIZE,T), "r")
    #Second nearest neighbor
    #f = h5py.File(FOLDER+"Ising2nd-%dx%d-%d-%.1f.hdf5"%(N,N,DSIZE,T), "r")
else:
    f = h5py.File(FOLDER+"potts-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,DSIZE,T), "r")    
train_data = f["Inputs"]
train_labels = f["Labels"]



# In[32]:


#Some code for building the RNN came from here:
#https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
#the 'self' variable is a relic of this code as it used to be in class PTBModel
#Empty class for the 'self' variable
class Empty():
    pass
def data_type():
    return tf.float32
def rnn_model_fn(features,labels,mode):
  """Function that creates the RNN model"""
  print (features['x'].shape[0])
  print (features)
  with tf.device(CONFIG.device):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    self = Empty()
    config = CONFIG
    self.batch_size = config.batch_size
    if mode == tf.estimator.ModeKeys.PREDICT:
        try:
            self.batch_size = int(features['x'].shape[0])
        except TypeError:
            if N < 5:
                self.batch_size = 2**(N**2-2)
            else:
                self.batch_size = 1024
    if mode == tf.estimator.ModeKeys.EVAL:
        self.batch_size = 1
    print(config.batch_size)
    self._is_training = is_training
    self._rnn_params = None
    self._cell = None
    self.num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    inputs = features['x']
    output, state = _build_rnn_graph(self,inputs, config, is_training)

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=logits)
    loss = tf.squared_difference(logits, labels)
    # Update the cost
    self._cost = tf.reduce_mean(loss)
    #cost is just mean squared error
    self._final_state = state
    total_loss = self._cost
    #Below is only for logging purposes
    #Looking at the cost isn't particularily useful in this case. (There is a minimum value dependant on the size of the grid)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('Train loss',total_loss)
        summary_hook = tf.train.SummarySaverHook(
            4,
            output_dir=DIR+"/eval",
            summary_op=tf.summary.merge_all())
        optimizer = tf.train.AdamOptimizer(learning_rate= config.learning_rate)
        train_op = optimizer.minimize(
            loss=total_loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op,training_hooks=[summary_hook])

    return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss)

def _build_rnn_graph(self, inputs, config, is_training):
    """Build the inference graph using LSTM or GRU cells."""
    #Dropout is untested
    def make_cell():
      cell = _get_cell(self,config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell
    #for multiple cells in paralell
    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    #Initialization of the RNN
    self._initial_state = cell.zero_state(self.batch_size, data_type())
    state = self._initial_state
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

def _get_cell(self, config, is_training):
    """Creates a basic lstm, block lstm, or gru cell as specified in the config class"""
    if config.rnn_mode == 'basic':
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == 'block':
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    if config.rnn_mode == 'gru':
        return tf.contrib.rnn.GRUCell(config.hidden_size)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)


# In[33]:


def st_dev(probs):
    """Returns average and standard deviation for values in an array"""
    avg = np.sum(probs)/len(probs)
    st_dv = np.power(np.sum([(x-avg)**2 for x in probs])/len(probs),0.5)
    return avg,st_dv
def _P(arrs):
    """Uses the RNN to output the probabilities for a set of NxN grids"""
    ins,outs = conv_inp(arrs)
    ins,outs = np.asarray(ins,dtype=np.float32),np.asarray(outs,dtype=np.float32)
    ising_predictor = tf.estimator.Estimator(
      model_fn=rnn_model_fn, model_dir=DIR)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": ins},batch_size= len(arrs),num_epochs=None,shuffle=False)
    preds = ising_predictor.predict(input_fn=predict_input_fn)
    final = []
    for idx in range(len(outs)):
        res = next(preds)
        if CONFIG.vocab_size ==1:
            final += [abs(np.prod(np.add(res,outs[idx])))/2**(N**2)]#Ising
        else:
            final += [np.prod([abs(np.sum(a)) for a in np.multiply(res,outs[idx])])/Q]#Potts
    return final

def dH(arrs1,arrs2):#these are in original format
    """Calculates actual and RNN probability based Energy differences for two sets of equal sized arrays"""
    if CONFIG.vocab_size ==1:
        P1 = _P(arrs1) #Ising
        P2 = _P(arrs2)
    else:
        P1 = _P([to_list(arrs) for arrs in arrs1])#Potts
        P2 = _P([to_list(arrs) for arrs in arrs2])
    dH = KB*T*(np.log(P1) - np.log(P2))
    real_dH = np.asarray([H(arr) for arr in arrs2])-np.asarray([H(arr) for arr in arrs1])
    print (real_dH)
    return dH,real_dH
def stats_(seed):
    """Outputs the average probaility of a configuration"""
    np.random.seed(seed)
    if CONFIG.vocab_size == 1:
        stats = st_dev(_P([domain(N,Q)for n in range(10000)]))
    else:
        stats = st_dev(_P([to_list(domain(N,Q))for n in range(10000)]))#potts
    print (stats)
    exp = np.math.ceil(-np.log10(stats[0]))
    print("%.3f ± %.3f (x10^%d)"%(stats[0]*10**exp,stats[1]*10**exp,-exp))
def res_plt(seed = 55):
    """Graphs actual Energy differences Vs RNN generated energy differences"""
    stats_(seed)
    ds = []
    ds = dH([domain(N,Q)for n in range(2000)],[domain(N,Q)for n in range(2000)])
    print(ds)
    abs_avg_err = np.sum(abs(ds[0]-ds[1]))/len(ds[0])
    mse = np.sum((ds[0]-ds[1])**2)/len(ds[0])
    rmse = mse**0.5
    print ('RMSE:', rmse)
    print ('CURRENT APPROXIMATE LOSS IS:', abs_avg_err)
    sys.stdout.flush()
    return abs_avg_err


# In[19]:



# In[20]:


def main(unused_argv):
    """Trains the RNN using train_data and train_labels"""
    global train_data,train_labels
    # Load training data
    k = 2 if CONFIG.vocab_size>1 else 1
    bsize=CONFIG.batch_size*(DSIZE*100000//k//CONFIG.batch_size)
    print(bsize)
    train_data_0 = np.asarray(train_data[:bsize],dtype=np.float32)
    train_labels_0 = np.asarray(train_labels[:bsize],dtype=np.float32)
    # Create the Estimator
    long_cfg = tf.estimator.RunConfig(keep_checkpoint_max = 500)# Retain the 200 most recent checkpoints.
    ising_predictor = tf.estimator.Estimator(
        model_fn=rnn_model_fn, model_dir=DIR,config=long_cfg)

    # Set up logging for predictions
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data_0},
        y=train_labels_0,
        batch_size=CONFIG.batch_size,
        num_epochs=1,
        shuffle=True)

    losses = [res_plt()]
    for x in range(CONFIG.EPOCHS):#number of epochs
        t = time.time()
        print("--------------------- epoch: ",x,"---------------------")
        for n in range(k):
            if k>1:
                train_data_0[:,:] = train_data[bsize*n:bsize*(n+1)]
                train_labels_0[:,:] = train_labels[bsize*n:bsize*(n+1)]
            ising_predictor.train(
                input_fn=train_input_fn,
                hooks=[logging_hook])
        print("RESULTS:")
        dt = (time.time()-t)
        print("%d:%d"%(dt//60,int(dt)-60*(dt//60)))
        losses+=[res_plt()]
    print(losses)
    res_plt()
if __name__ == "__main__":
    try:
        tf.app.run()
    except SystemExit:
        print("done")




