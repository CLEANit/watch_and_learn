from metrics import *
import h5py
"""This is here to compute the loss on a uniform distribution, the workflow is as follows:
dataset(filename,bins,*args) -> makes finds (default 1000*Q) configurations for each bin and saves the bins to an h5py file
pred_bins(model.Hs,filename,bins) -> goes through a generated file with bins and predicts with a hamiltonian
NOTE: bin_compare ONLY works with bins generated with tol ~ 0
bin_compare(bins,predictions) -> compares outputs with the bins
rebin_compare(bins,ans1,ans2) remakes bins according to the values in ans1(predictions) and compares them to ans2"""
def st_dev(probs):
    """returns the average, sum, and mean squared value of the probs distribution"""
    avg = np.sum(probs)/len(probs)
    st_dv = np.power(np.sum([(x-avg)**2 for x in probs])/len(probs),0.5)
    mse=np.sum([(x)**2 for x in probs])/len(probs)
    return avg,st_dv,mse
def find_e(e,tol=1e-6,p=0.2):
   """Finds an arbitrary state with an energy within margins tol of energy e"""
   d = domain(N,Q)
   Hi = H(d)
   i=0
   while True:
       d2 = augment(d)
       Hf=H(d2)
       if abs(Hf-e)<abs(Hi-e) or np.random.random()<p:
           d,Hi=d2,Hf
       if abs(Hi-e)<tol:
           yield d
           d = augment(d)#augment(augment(augment(d)))
           Hi = H(d)
       i+=1
def dataset(filename,bins,tol,perbin=1000,p=0.05):
    """creates a set of states at given energies"""
    with h5py.File(filename, "w") as file:
        for b in bins:
            print("\n",b)
            arrs=[]
            f = find_e(b,tol,p)
            for wah in range(perbin):
               a=np.asarray(next(f))
               if MODEL!="POTTS1":
                   arrs += [a,a*-1]
               else:
                   arrs += [(a+r)%Q for r in range(Q)]
               if wah%(perbin//100)==0:
                   print(wah/perbin,end = "|")
            file.create_dataset(str(b), data=arrs)
#bins=[x for x in range(-128,129,4) if abs(x)!=124]
#dataset("Uniform2.h5py",bins,1e-6,perbin=4000) 
def pred_bins(Hs,filename,bins):
    """Opens a dataset and makes predicitons with Hs"""
    p=0.05
    binpreds = []
    with h5py.File(filename, "r") as file:
        for b in bins:
            print(b,end=" ")
            data = file[str(b)]
            binpreds+=[Hs(np.asarray(data))]
    return binpreds
def rebin_compare(bins,ans1,ans2,tol):
    """Compares two sets of predictions and returns an xy graph with errors"""
    x=[[] for n in range(len(bins))]
    y=[[] for n in range(len(bins))]
    for i in range(len(ans1)):
        for j in range(len(ans1[0])):
            for k in range(len(bins)):
                if abs(ans1[i][j]-bins[k])<tol:
                    x[k]+=[ans1[i][j]]
                    y[k]+=[ans2[i][j]]
                    #break
    dist = [[],[],[]]
    mse=0
    total=0
    for i in range(len(bins)):
        if len(y[i])<2:
            continue
        info=st_dev(np.asarray(y[i])-np.asarray(x[i]))
        dist[0]+=[i]
        dist[1]+= [info[0]]
        dist[2]+=[info[1]]
        mse+=info[2]*len(x[i])
        total+= len(x[i])
    print("RMSE:",(mse/total)**0.5)
    return dist
def bin_compare(bins,ans1):
    """Compares a set of predictions to the bin labels and returns xy graph w errors"""
    dist = [bins,[],[]]
    mse=0
    total=0
    for i in range(len(bins)):
        info=st_dev(np.asarray(ans1[i])-bins[i])
        dist[1]+= [info[0]]
        dist[2]+=[info[1]]
        mse+=info[2]*len(ans1[i])
        total+= len(ans1[i])
    print("RMSE:",(mse/total)**0.5)
    return dist
__otherconst = set_constants
def set_constants(n=8,q=2,kb=1.0,seed=12,model="ISING1"):
   global N,Q,KB,G,MODEL
   N=n
   Q=q
   KB=kb
   MODEL=model
   __otherconst(n,q,kb,seed,model)
