import numpy as np
#from matplotlib import pyplot as plt
import time
import tensorflow as tf
import sys

f=2
c=1
def NN(_in):
   """The network to be run on each tile"""
   tile_size = f + 2*c
   _in = tf.reshape(_in, (-1, tile_size**2))
   nn = tf.contrib.layers.fully_connected(_in, 32)
   nn = tf.contrib.layers.fully_connected(nn, 64)
   nn = tf.contrib.layers.fully_connected(nn, 1, activation_fn=None)
   return nn
class L_EDNN():
    def __init__(self,SAVEDIR,N=8,Q=2):
        """Initialize the ednn"""
        L=N
        self.Q=Q
        self.x = tf.placeholder(tf.float32, (None, 4, 4), name='input_image')
        self.predicted=NN(self.x) #= tf.reduce_sum(output, axis=1)
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

def break_apart(arr):
    arr2 = np.asarray(arr)
    x,y=arr2.shape
    arr3 = np.pad(arr2,((1,1),(1,1)),mode="wrap")
    arr4=np.asarray([arr3[a:a+4,b:b+4] for b in range(0,y,2) for a in range(0,x,2)])
    return arr4

def affected(n,y,x):
    bx=x//2
    by=y//2
    sx=x%2
    sy=y%2
    c=n//2
    s1= [by+bx*c,sy+1,sx+1]
    s2= [(by+2*sy-1)%(n//2)+bx*c,(sy-1)%4,sx+1]
    s3= [by+(bx+2*sx-1)%(n//2)*c,sy+1,(sx-1)%4]
    s4=[(by+2*sy-1)%(n//2)+(bx+2*sx-1)%(n//2)*c,(sy-1)%4,(sx-1)%4]
    return s1,s2,s3,s4
#############################################################################
def domain(n,q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be formatted for the rnn but not the hamiltonians"""
    return [[np.random.randint(0,2)*2-1 for a in range(n)]for b in range(n)]#ising

def prep(arr,i,j):
    flip(arr,i,j)
    a = atable[i][j][0]
    val = arr[a[0]][a[1]][a[2]]
    prep=[arr[atable[i][j][k][0]] for k in range(4)]
    return val,prep
def dh(change,partials,arr,i,j):
    diff=0
    partials_2 = partials.copy()
    for k in range(4):
        k2=atable[i][j][k][0]
        diff+=change[k]-partials[k2]
        partials_2[k2] = change[k]
    return diff,partials_2
def flipsite(shape,l=None):
    """Spin flip where a random spin in the grid is changed (Potts is still an integer representation)"""
    x,y = shape
    x_ = np.random.randint(0,x)
    y_ = np.random.randint(0,y)
    return x_,y_
def flip(arr,i,j):
    for a in atable[i][j]:
        arr[a[0]][a[1]][a[2]]*=-1
def metropolis(Hs,n,B,iterate = 2500,keep=0.2):
   width=len(B)
   iterate= int(iterate)
   """Runs a monte carlo simulation with the specified hamiltonian function H"""
   C = np.e**(-B)
   arrs = [domain(n,n) for a in range(width)]
   shape=np.shape(arrs[0])
   size=np.prod(shape)
   arrs=[break_apart(arr) for arr in arrs]
   partials=[Hs(arr) for arr in arrs]
   counter=0
   ct=0
   U=np.zeros(width)
   for i in range(iterate):
       sites=[flipsite(shape)for a in range(width)]
       vals=[]
       preps=[]
       for j in range(width):
          pr=prep(arrs[j],*sites[j])
          vals+=[pr[0]]
          preps+=pr[1]
       H_eval=Hs(preps)
       for j in range(width):
          dH,partials_2 = dh(H_eval[4*j:4*(j+1)],partials[j],arrs[j],*sites[j])

          n=np.random.random()
          if dH <=0 or n < C[j]**dH:
              partials[j]=partials_2
          else:
             flip(arrs[j],*sites[j])

          if i>iterate*(1-keep) and (i%10)==0:
              U[j]+=sum(partials[j])/size
              ct+=(j==0)
       if i>counter:
          counter+=iterate*0.01
          print("%.0f"%(100*i/iterate),end="%|")
          sys.stdout.flush()
   return U/ct
def errspinsQueue(B,width,steps,seed):
   model = L_EDNN(FILENAME)
   Hs=model.Hs
   np.random.seed(seed)
   steps = int(steps)
   x = [B for z in range(width)]
   res = [x,[],[]]
   t=time.time()
   res=metropolis(Hs,N,np.asarray(x),steps,keep=0.8)
   f_avg = np.sum(res)/width
   f_st_dv = np.power(np.sum([(i-f_avg)**2 for i in res])/width,0.5)
   print()
   print(f_avg,f_st_dv)
   print(res)
   return res

#####################################################################
kwargs = {}
for arg in sys.argv[1:]:
    name,val = arg.split("=")
    try:
        kwargs[name] = val
        kwargs[name] = float(val)
        kwargs[name] = int(val)
    except:
        pass

old_stdout = sys.stdout

if "file" in kwargs:
   FILENAME=kwargs["file"]
else:
   FILENAME="finalEDNNISING"

log_file = open(FILENAME.split("EDNN")[0]+
                "/U-%dx%d-%f.log"%(kwargs["N"],kwargs["N"],kwargs["B"]),"w")


sys.stdout = log_file

#kwargs={"N":16,"Ti":5,"Tf":50,"div":10.0,"perspin":100,"seed":5}


N=kwargs["N"]#16
atable = [[affected(N,y,x) for x in range(N)] for y in range(N)]
print(N)
print(kwargs)
t=time.time()
#res= errspinsQueue(5,50,10,2e2*8**2,5)
res= errspinsQueue(kwargs["B"],kwargs["W"],kwargs["perspin"]*N**2,kwargs["seed"])
print(time.time()-t)

sys.stdout = old_stdout

log_file.close()
