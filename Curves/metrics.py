import numpy as np
from matplotlib import pyplot as plt
import sys
import h5py
"""THIS FILE CONTAINS MOST METRICS FOR TESTING THE NETWORKS
IMPORTANT:
    rmse(model.Hs): gets a basic root mean squared error on a random distribution of configurations (using a networks Hs funct)
    set_constants(*kwargs): sets the constants in the module to ensure youre testing for the right nidel and etc
    logtransition(model.Hs,starting_temp,ending_temp,temp_divisor,steps,seed,keep,fn,start): makes phase transition graph saved to a file
    """
def domain(n,q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be formatted for the rnn but not the hamiltonians"""
    if MODEL =="POTTS1":
        return [[np.random.randint(q) for a in range(n)]for b in range(n)]#potts
    else:
        return [[np.random.randint(0,2)*2-1 for a in range(n)]for b in range(n)]#ising
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
class SpinGlass():
    """ Handles nearest neighbor spinglass hamiltonians"""
    def __init__(self,seed,N):
        self.N = N
        np.random.seed(seed)
        self.h = np.random.normal(0,1,[N,N])
        self.v = np.random.normal(0,1,[N,N])  
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
        ad = np.asarray([a for a in range(1,self.N**2+1)]).reshape([N,N])
        ah = np.roll(ad,1,axis=1)
        av = np.roll(ad,1,axis=0)
        s1 = "".join(["%d %d %f\n"%(ad[i][j],ah[i][j],self.h[i][j]) for i in range(self.N) for j in range(self.N)])
        s2 = "".join(["%d %d %f\n"%(ad[i][j],av[i][j],self.v[i][j]) for i in range(self.N) for j in range(self.N)])
        return s1+s2
    def dh(self,shape,arr,i,j):
        y,x = shape
        d=arr[i-1][j]*self.v[i][j]
        u=arr[(i+1)%y][j]*self.v[(i+1)%y][j]
        l=arr[i][j-1]*self.h[i][j]
        r=arr[i][(j+1)%x]*self.h[i][(j+1)%x]
        return arr[i][j]*(d+u+l+r)*2
class SpinGlass2():
    """Handles all interacting spinglass hamiltonians"""
    def __init__(self,seed,N):
        self.N = N**2
        np.random.seed(seed)
        self.c= np.random.normal(0,1,[self.N]*2)
        self.c2 = self.c+0
        for i in range(self.N):
            for j in range(self.N):
                if j<=i:
                    self.c[i][j]=0
                    self.c2[i][j]=0
                if j<i:
                    self.c2[i][j]=self.c2[j][i]
    def H(self,arr):
        a=np.reshape(arr,[self.N,1])
        b=np.reshape(arr,[1,self.N])
        return -np.sum(np.multiply(self.c,np.matmul(a,b)))
    def dh(self,shape,arr,i,j):
        b=np.reshape(arr,[self.N])
        return arr[i][j]*2*np.sum(np.multiply(self.c2[i*shape[0]+j],b))
    def __str__(self):
        string=""
        for i in range(self.N):
            for j in range(self.N):
                if self.c[i][j]!=0:
                    string+= "%d %d %f\n"%(i+1,j+1,self.c[i][j])
        return string
def H(arr):
    """Calculates a hamiltonian for all models"""
    if MODEL[:-1]=="SPINS":
        return G.H(arr)
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
def rmse(Hs,seed = 55,plot =True):
    """Graphs actual Energy differences Vs RNN generated energy differences"""
    np.random.seed(seed)
    ds = []
    ds = dH(Hs,[domain(N,Q)for n in range(2000)],[domain(N,Q)for n in range(2000)])
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

def paralellmetro(Hs,N,y,T,iterate = 2500):
    """Runs a monte carlo simulation with the specified hamiltonian function Hs"""
    C = np.e**(-1.0/(KB*T))
    arrs = [domain(N,y) for idx in range(len(T))]
    Hi = Hs(arrs)
    m= [[metric(arr)] for arr in arrs]
    for it in range(iterate):
        arr2 = [augment(arr) for arr in arrs]
        Hf = Hs(arr2)
        for i in range(len(T)):
            dH = Hf[i]-Hi[i]
            n=np.random.random()
            if dH <=0 or n < C[i]**dH:
                Hi[i] = Hf[i]
                arrs[i] = arr2[i]
                m[i]+= [metric(arrs[i])]
            else:
                m[i]+=[m[i][-1]]
        if (it%(iterate//100))==0:
            print("%.2f"%(it/iterate*100),end="% ")
            sys.stdout.flush()

    return m
def phasetransition(Hs,num1,num2,div,steps,seed):
    """Wrapper for paralell metro"""
    np.random.seed(seed)
    steps = int(steps)
    #q.put("Started %d"%seed)
    ts = np.asarray([z/div for z in range(num1,num2)])
    res = [ts,[],[],seed]
    metro = paralellmetro(Hs,N,N,ts,steps)
    for n in metro :
        eq = n[-steps//5:]
        avg = np.sum(eq)/len(eq)
        st_dv = np.power(np.sum([(x-avg)**2 for x in eq])/len(eq),0.5)
        res[1]+=[avg]
        res[2]+=[st_dv]
    #q.put(res)
    print()
    return res
def spinPotts(arr):
   """Returns the average spin magnitude of a potts grid"""
   cos = np.sum([[np.cos(n*2*np.pi/Q)for n in m]for m in arr])/(len(arr)*len(arr[0]))
   sin = np.sum([[np.sin(n*2*np.pi/Q)for n in m]for m in arr])/(len(arr)*len(arr[0]))
   return (sin**2+cos**2)**0.5
def spinIsing(arr):
    """Returns the average spin of an ising grid"""
    return np.sum(arr)/np.prod(np.shape(arr))
def metric(arr):
    if MODEL == "ISING1":
        return abs(spinIsing(arr))
    elif MODEL=="POTTS1":
        return abs(spinPotts(arr))
    else:
        return G.H(arr)
def isingmetro(Hs,N,y,T,iterate = 2500):
   """Runs a monte carlo simulation optimised for the ising model (const ram)"""
   C = np.e**(-1.0/(KB*T))
   arrs = [domain(N,y) for idx in range(len(T))]
   Hi = Hs(arrs)
   m = [{a:0 for a in range(0,N**2+1)} for arr in arrs]
   for i in range(len(T)):
       m[i][abs(np.sum(arrs[i]))]=1
   for it in range(iterate):
       arr2 = [augment(arr) for arr in arrs]
       Hf = Hs(arr2)
       for i in range(len(T)):
           dH = Hf[i]-Hi[i]
           n=np.random.random()
           if dH <=0 or n < C[i]**dH:
               Hi[i] = Hf[i]
               arrs[i] = arr2[i]
           if it>2*iterate//5:
               m[i][abs(np.sum(arrs[i]))]+=1
       if (it%(iterate//100))==0:
           print("%.2f"%(it/iterate*100),end="% ")
           sys.stdout.flush()
   return m
def isingtransition(Hs,num1,num2,div,steps,seed,keep=0.2):
   np.random.seed(seed)
   steps = int(steps)
   #q.put("Started %d"%seed)
   ts = np.asarray([z/div for z in range(num1,num2)])
   res = [ts,[],[],seed]
   metro = isingmetro(Hs,N,N,ts,steps)
   for n in metro :
       eq = n
       #print(eq)
       sum_=np.sum([eq[i] for i in eq])
       #print(sum_)
       #avg = np.sum(eq)/len(eq)
       avg = np.sum([i*eq[i] for i in eq])/sum_/N**2
       st_dv = np.power(np.sum([eq[i]*(i/N**2-avg)**2 for i in eq])/sum_,0.5)
       res[1]+=[avg]
       res[2]+=[st_dv]
   #q.put(res)
   print()
   return res
def logmetro(Hs,N,y,T,file_name,iterate = 2500,start=0):
   """Runs a monte carlo simulation with the output saved"""
   C = np.e**(-1.0/(KB*T))
   if start==0:
       arrs = [domain(N,y) for idx in range(len(T))]
       m= [[metric(arr)] for arr in arrs]
   else:
       f = h5py.File(file_name,"r")
       arrs=np.asarray(f["States"])
       m = [[] for arr in arrs]
       f.close()
   M= [metric(arr) for arr in arrs]
   Hi = Hs(arrs)
   for it in range(start,iterate):
       arr2 = [augment(arr) for arr in arrs]
       Hf = Hs(arr2)
       for i in range(len(T)):
           dH = Hf[i]-Hi[i]
           n=np.random.random()
           if dH <=0 or n < C[i]**dH:
               Hi[i] = Hf[i]
               arrs[i] = arr2[i]
               M[i]=metric(arrs[i])
           m[i]+=[M[i]]
       if (it%(iterate//100))==0:#can be something else
           if start==0:
               print("started")
               f = h5py.File(file_name, "w")
               f.create_dataset("States", data=arrs)
               f.create_dataset("Spins", data=m, chunks=True,maxshape=(len(T),iterate))
               print(f["Spins"])
               f.close()
               start=1
           else:
               try:
                   f = h5py.File(file_name, "a")
                   #print(f["Spins"])
                   f["Spins"].resize((f["Spins"].shape[1] + len(m[0])), axis = 1)
                   f["Spins"][:,-len(m[0]):] = m
                   f["States"][:,:] = arrs
                   m = [[] for arr in arrs]
                   f.close()
               except Exception as e:
                   print(e)
                   print(f.keys())
                   f.close()
                   return
           print("%.2f"%(it/iterate*100),end="% ")
           sys.stdout.flush()
   f= h5py.File(file_name, "r")
   return f["Spins"]
def logtransition(Hs,num1,num2,div,steps,seed,keep=0.2,fn="test.h5py",start=0):
   """Makes a phase transition using the params as follows:
   temperature ranges from num1/div to num2/div with a dot at each 1/div, only keep monte carlo metrics are used, 
   and it runs for 'steps' monte carlo steps. It's saved to the file specified in fn, and if start>0 it will load
   the file at fn and assume the simulation resumes from the number start"""
   np.random.seed(seed)
   steps = int(steps)
   #q.put("Started %d"%seed)
   ts = np.asarray([z/div for z in range(num1,num2)])
   res = [ts,[],[],seed]
   metro = logmetro(Hs,N,N,ts,fn,steps,start=start)
   metro=np.asarray(metro)
   for n in metro :
       eq = n[-int(steps*keep):]
       avg = np.sum(eq)/len(eq)
       st_dv = np.power(np.sum([(x-avg)**2 for x in eq])/len(eq),0.5)
       res[1]+=[avg]
       res[2]+=[st_dv]
   #q.put(res)
   print()
   return res
def get_graph(num1,num2,div,steps=None,keep=0.2,fn="test.h5py"):
   """Pulls a graph from logmetro"""
   print("starting")
   f= h5py.File(fn, "r")
   metro = f["Spins"]
   print("opened file")
   #metro=np.asarray(metro)
   if steps==None:
       steps=metro.shape[1]
       print(steps)
   steps = int(steps)
   #q.put("Started %d"%seed)
   ts = np.asarray([z/div for z in range(num1,num2)])
   res = [ts,[],[],0]
   for n in range(metro.shape[0]):
       print(n,end=" ")
       eq = np.asarray(metro[n][-int(steps*keep):])
       avg = np.sum(eq)/len(eq)
       st_dv = np.power(np.sum([(x-avg)**2 for x in eq])/len(eq),0.5)
       res[1]+=[avg]
       res[2]+=[st_dv]
   f.close()
   #q.put(res)
   print()
   return res
def workpls():
    print("ok")
def set_constants(n=8,q=2,kb=1.0,seed=12,model="ISING1"):
    global N,Q,KB,G,MODEL
    N=n
    Q=q
    KB=kb
    MODEL=model
    if model=="SPINS2":
        G = SpinGlass2(seed,n)
    else:
        G = SpinGlass(seed,n)
def print_constants():
    global N,Q,KB,G,MODEL
    print("N=%s Q=%s KB=%s MODEL=%s"%(N,Q,KB,MODEL))
set_constants()
