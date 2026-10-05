"""Independent NumPy/Python rules oracle used by differential tests."""
import numpy as np

def classify(counts):
    c=np.asarray(counts); total=int(c.sum()); out=set()
    if np.any(c<0) or np.any(c[:13]>4) or np.any(c[13:]>1): return out
    def put(t,r,n=1): out.add((t,r,n))
    for r in range(15):
        if c[r]==total:
            if total==1: put('single',r)
            if total==2: put('pair',r)
            if total==3: put('trio',r)
            if total==4: put('bomb',r)
    if total==2 and c[13]==c[14]==1: put('rocket',14)
    for r in range(13):
        rem=c.copy(); rem[r]=0
        if c[r]==3:
            if total==4: put('trio1',r)
            if total==5 and np.count_nonzero(rem==2)==1: put('trio2',r)
        if c[r]==4:
            if total==6 and not (rem[13] and rem[14]): put('four2',r)
            if total==8 and np.count_nonzero(rem==2)==2: put('four22',r)
    for start in range(12):
        for end in range(start+1,13):
            n=end-start; chunk=c[start:end]; rem=c.copy(); rem[start:end]=0
            if n>=5 and total==n and np.all(chunk==1): put('straight',end-1,n)
            if n>=3 and total==2*n and np.all(chunk==2): put('pairs',end-1,n)
            if n>=2 and np.all(chunk==3):
                if total==3*n: put('plane',end-1,n)
                if total==4*n and np.max(rem)<=3 and not (rem[13] and rem[14]): put('plane1',end-1,n)
                if total==5*n and np.all((rem==0)|(rem==2)) and np.count_nonzero(rem==2)==n: put('plane2',end-1,n)
    return out

def beats(a,b):
    if b is None: return True
    ta,ra,na=a; tb,rb,nb=b
    if ta=='rocket': return tb!='rocket'
    if ta=='bomb' and tb not in ('bomb','rocket'): return True
    return ta==tb and na==nb and ra>rb

def score(landlord,winner,bid,bombs,plays):
    won=landlord==winner
    spring=won and all(plays[i]==0 for i in range(3) if i!=landlord)
    anti=not won and plays[landlord]==1
    amount=bid*(1+bombs+int(spring or anti))*(1 if won else -1)
    return np.array([2*amount if i==landlord else -amount for i in range(3)],np.float32)
