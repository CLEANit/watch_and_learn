from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py
from RNN_MODELS import spiral
"""THIS FILE IS FOR RUNNING RNN MODELS. Here are the important functions:
    RNN.Hs(arrs): for getting predictions
    RNN.training(metric_funct,*args,**kwargs): for making loss curves
    RNN.gen_data(dir_,div): for generating ednn train data"""
#tf.enable_eager_execution()
from tensorflow.python.client import device_lib
 # temperature, in units of kB (defined to be 1 below)
Q = 2  # Potts model, number of angles
KB = 1.0
N = 8  # NxN spin model
class Config():
    __Ising={
    'learning_rate': 0.0005,
    'num_layers' : 1,
    'num_steps' : N**2 - 1,
    'N':N,
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
    "vocab_size" : 3,
    "DSIZE":256,
    }
    def __init__(self,ispotts,**kwargs):
        """Uses the parameters in kwargs as config. if an argument is not specified, the default for said model is used"""
        self.__dict__.update(Config.__Ising)
        if ispotts:
            self.__dict__.update(Config.__Potts)
        self.__dict__.update(kwargs)
        global T,Q,N
        N=self.N
        self.num_steps=N**2 - 1
        T=self.T
        print("T=%s, N=%s"%(T,N))
        if ispotts and Q !=self.vocab_size:
            Q=self.vocab_size
            print("Set Q to %d"%Q)
        spr=spiral(np.arange(N**2).reshape(N,N))
        global indices
        indices=np.asarray([spr.index(a) for a in range(N**2)])

def unspiral(arr):
    return np.asarray(arr)[indices].reshape(N,N)

def one_by_one(features,config,state,reuse):
    with tf.device(config.device):
        inputs = features
        size = config.hidden_size
        vocab_size = config.vocab_size
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUCell(config.hidden_size)], state_is_tuple=True)
        #cell = tf.contrib.rnn.GRUCell(config.hidden_size)
        output, state = use_cell(inputs,cell,state, config, reuse)
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
         # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, tf.shape(inputs))
        return logits,state
def use_cell(inputs,cell,state, config, reuse):
    """Build the inference graph using LSTM or GRU cells."""
    outputs = []
    with tf.variable_scope("RNN"):
        #state has to be in [] because it is a multiRNN cell of size 1
        cell_output, state = cell(inputs[:, 0, :], [state])
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    #returning state[0] because it is a size 1 tuple
    return output, state[0]
class URNN():
    def generate_data(self,width=1,seed=21,show_pred=False):
        """Returns tuple (Lattices, Energies, Probabilities ) of lattices it generates"""
        np.random.seed(seed)
        cur_state = np.zeros([width, self.CONFIG.hidden_size])
        total=[]
        vect = np.random.randint(0,2,[width,1,1])*2-1
        #print(vect)
        #print("doing stuff")
        preds=np.zeros([self.CONFIG.num_steps,width])
        choices=np.zeros([self.CONFIG.num_steps+1,width])
        choices[0]=vect.reshape(width)
        for x in range(self.CONFIG.num_steps):
            #for ising pred should be a float in [-1,1]
            pred,cur_state = self.sess.run([self.logits,self.state],
                {self.features:vect,self.initial_state:cur_state})
            preds[x]=pred.reshape(width)
            vect=np.zeros([width,1,1])
            rnd=np.random.random([width])
            for w_i in range(width):
                vect[w_i][0][0]=1 if (rnd[w_i]*2-1)<preds[x][w_i] else -1
            choices[x+1]=vect.reshape(width)
        #print(preds,choices)
        val = abs(np.prod(np.add(choices[1:],preds),axis=0))/2**(N**2)
        #print(val)
        #abs(np.prod(np.add(res,outs[idx])))/2**(N**2)
        val=np.asarray(val,dtype=np.float64)
        #print([val])
        E=KB*T*(-np.log(val))+self.C
        #print()
        if show_pred:
            return [unspiral(choices[:,i]) for i in range(width)],E,val,preds
        return [unspiral(choices[:,i]) for i in range(width)],E,val
    
    def __init__(self,filename,cbatch=[10000],checkpoint=None,ref=None,**kwargs):
        """creates the rnn graph from a pointer file, or kwargs if the fully define the rnn"""
        global N
        #8x8 transitions at about 2.6, 4x4 at 4.0
        self.ckpt=checkpoint
        print(self.ckpt)
        if len(kwargs)>0:
            global CODE, FLATTEN, MODEL
            CODE, FLATTEN, MODEL = kwargs["code"],kwargs["flat"],kwargs["model"]
            del kwargs["code"]
            del kwargs["flat"]
            del kwargs["model"]
            self.w_dir=filename
        else:
            kwargs = self.parse_file(filename)
        self.CONFIG = Config(MODEL=="POTTS1",**kwargs)
        self.DIR = self.w_dir+(CODE+MODEL)
        config = self.CONFIG
        #self.new_session()
        self.features = tf.placeholder(tf.float32, [None,1,config.vocab_size])
        self.initial_state= tf.placeholder(tf.float32, [None, config.hidden_size])
        self.logits,self.state = one_by_one(self.features,config,self.initial_state,True)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.ckpt!=None:
            self.saver.restore(self.sess,self.DIR+'/model.ckpt-%d'%self.ckpt)
        else:
            fil = open(self.DIR+"/checkpoint")
            self.saver.restore(self.sess,self.DIR+'/'+fil.readline().split('"')[-2])
            fil.close()
        self.C = 0
    def setC(self,num_vals=10000):
        vals=self.generate_data(width=num_vals,seed=None)[1]
        avg=np.sum(vals)/num_vals
        print(self.C,avg)
        self.C-=avg
    def gen_data(self,dir_,div = 240):
        """Uses the RNN to generate traindata and labels for the ednn"""
        length = 8000
        f = h5py.File(dir_+"EDNN-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,div,T), "w")
        for i in range(div):
            in_,out,_=self.generate_data(width=length,seed=None)
            in_ = np.asarray(in_,dtype=np.float32)  
            out = np.asarray(out,dtype=np.float32)
            if i==0:
                f.create_dataset("Inputs", data=in_, chunks=True,maxshape=(length*div, N,N))
                f.create_dataset("Labels", data=out, chunks=True,maxshape=[length*div])
                f.close()
                f = h5py.File(dir_+"EDNN-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,div,T), "a")
            else:
                f["Inputs"].resize((f["Inputs"].shape[0] + length), axis = 0)
                f["Inputs"][-length:] = in_
                f["Labels"].resize((f["Labels"].shape[0] + length), axis = 0)
                f["Labels"][-length:] = out
            print("----------------------------------",(i+1)/div*100,"----------------------------------")
        f.close()
    def entropy(self,width=10000,seed=None):
        _,__,p = self.generate_data(width,seed)
        return np.sum(-np.log(p))/width
    def close(self):
        """resets all tf variables"""
        tf.reset_default_graph()
        self.sess.close()
        
