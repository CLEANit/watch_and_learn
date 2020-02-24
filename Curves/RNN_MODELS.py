from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py

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
        global Q,T
        if self.vocab_size>1:
            Q=self.vocab_size
        T=self.T
def to_list(arr):
    """Creates a Q dimension vector with 0s except for the index of the spin for each spin in the potts grid"""
    def rndarr(Q,i):
        arr = [0]*Q
        arr[i] = 1
        return arr
    return [[rndarr(Q,i) for i in a] for a in arr]
def domain(n,q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be formatted for the rnn but not the hamiltonians"""
    if MODEL =="POTTS1":
        return [[np.random.randint(q) for a in range(n)]for b in range(n)]#potts
    else:
        return [[np.random.randint(0,2)*2-1 for a in range(n)]for b in range(n)]#ising
#TODO: define what this function does or is for  
def snake(arr):
    """flattens a 2d array into a 1d in a snake pattern """
    snake_ = []
    n = 1
    for a in arr:
        snake_ += list(a[::n])
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

#Some code for building the RNN came from here:
#https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py

def data_type():
    return tf.float32

def format_d(arrs):
    """Formats a potts input for use with the EDNN (spins are squished from -1 to 1)"""
    return [[[2*x/(Q-1) - 1 for x in y]for y in arr]for arr in arrs]

class RNN():
    def rnn_model_abridged(self,inputs,config,mode=tf.estimator.ModeKeys.EVAL):
      """Function that creates the RNN model this can be used with sess.run"""
      with tf.device(config.device):
        print(self.batch_size)
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        self._is_training = is_training
        self._rnn_params = None
        self._cell = None
        self.num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        output, state = self._build_rnn_graph(inputs, config, is_training)
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
         # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        return logits

    def _build_rnn_graph(self, inputs, config, is_training):
        """Build the inference graph using LSTM or GRU cells."""
        #Dropout is untested
        def make_cell():
          cell = self._get_cell(config, is_training)
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
    def _P(self,arrs):
        """Uses the RNN to output the probabilities for a set of NxN grids"""
        ins,outs = conv_inp(arrs)
        ins,outs = np.asarray(ins,dtype=np.float32),np.asarray(outs,dtype=np.float32)
        #print(ins.shape)
        preds = self.sess.run([self.outputs],feed_dict={self.inputs: ins})
        final = []
        for idx in range(len(outs)):
          #print(len(preds[0]))
          res = preds[0][idx]
          if self.CONFIG.vocab_size ==1:
              final += [abs(np.prod(np.add(res,outs[idx])))/2**(N**2)]#Ising
          else:
              final += [np.prod([abs(np.sum(a)) for a in np.multiply(res,outs[idx])])/Q]#Potts
        return final

    def Hs(self,arrs1):
        """Calculates a hamiltonian using the rnn for a set of grids"""
        if self.batch_size != len(arrs1):
            self.sess.close()
            self.new_session(len(arrs1))
        if MODEL == "POTTS1":
            P1 = self._P([to_list(arrs) for arrs in arrs1])
        else:
            P1 = self._P(arrs1)
        H = KB*T*(-np.log(P1))+self.C
        return H
    def H(self,arr):
        """Calculates the hamiltonian using the rnn for a grid"""
        return self.Hs([arr])
    def new_session(self,batch_size=2000):
        """Resets and remakes the rnn graph, restoring the checkpoint stored in self.ckpt
        This is necessary to change the batch size."""
        self.batch_size = batch_size
        #self.sess.close()
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, (None, N**2-1, self.CONFIG.vocab_size), name='input_image')
        self.outputs = self.rnn_model_abridged(self.inputs,self.CONFIG,mode=batch_size)
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        if self.ckpt!=None:
            self.saver.restore(self.sess,self.DIR+'/model.ckpt-%d'%self.ckpt)
        else:
            fil = open(self.DIR+"/checkpoint")
            self.saver.restore(self.sess,self.DIR+'/'+fil.readline().split('"')[-2])
            fil.close()
    def training(self,metric_funct,*args,**kwargs):
        """runs a metric function with all models specified in checkpoint. This is used to make loss curves
        NOTE: args and kwargs are passed to the metric function"""
        #metric must have constant input size
        fil = open(self.DIR+"/checkpoint")
        metrics=[]
        steps = []
        metric_funct(self.Hs,*args,**kwargs)
        f_num = float(fil.readline().split('"')[-2].split("-")[-1])
        for info in fil:
            print(info.split('"')[-2])
            self.saver.restore(self.sess,self.DIR+'/'+info.split('"')[-2])
            metrics+=[metric_funct(self.Hs,*args,**kwargs)]
            steps+=[float(info.split('"')[-2].split("-")[-1])]
        self.new_session()
        return [steps,metrics]
    def parse_file(self,filename):
        """Builds dictionary values from command line arguments specified in a file (generally sh file)"""
        global CODE, FLATTEN, MODEL
        for fil in open(filename,"r"):
            argv = fil.strip().split(" ")
            if argv[0] == "###":
                self.w_dir = argv[1]
        CODE, FLATTEN, MODEL = argv[2:5]
        kwargs = {}
        for arg in argv[5:]:
            name,val = arg.split("=")
            try:
                kwargs[name] = val
                kwargs[name] = float(val)
                kwargs[name] = int(val)
            except:
                pass
        return kwargs
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
        self.new_session()
        self.C = 0
        self.ref = ref
        if ref != None:
            self.C = -np.sum(self.Hs([ref]))
        else:
            self.C = -np.sum([self.Hs([domain(N,Q) for x in range(a)]) for a in cbatch])/np.sum(cbatch)
    def close(self):
        """resets all tf variables"""
        tf.reset_default_graph()
    def gen_data(self,dir_,div = 200):
        """Uses the RNN to generate traindata and labels for the ednn"""
        length = 8000
        f = h5py.File(dir_+"EDNN-%dx%d-Q%d-%d-%.1f.hdf5"%(N,N,Q,div,T), "w")
        for i in range(div):
            in_ = [domain(N,Q) for x in range(length)]
            out = self.Hs(in_)
            if MODEL == "POTTS1":
                in_ = np.asarray(format_d(in_),dtype=np.float32)  
            else:
                in_ = np.asarray(in_,dtype=np.float32)  
            print (in_[0])
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
