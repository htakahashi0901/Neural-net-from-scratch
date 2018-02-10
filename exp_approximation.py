import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import time
import copy
%matplotlib inline
from IPython.core.debugger import Tracer
class Network(object):
    def __init__(self,size):
        #self.w=np.array([np.loadtxt(i) for i in glob.glob(r'./trained/w*.txt')])
        #self.b=np.array([np.loadtxt(i) for i in glob.glob(r'./trained/b+.txt')])
        self.w=np.array([np.random.randn(j,i) for i,j in zip(size[:-1],size[1:])])
        self.b=np.array([np.random.randn(i,1) for i in size[1:]])
        
        self.batch=100
        self.eta=0.01
        self.size=size
        self.alpha=0.1
    
    def get_tx(self,n):
        x=np.linspace(0,1,n).reshape(1,n)
        y=np.exp(x)
        return x,y
    
    def forward(self,w,x,b):
        z=np.matmul(w,x)+b
        self.l_z.append(z)
        buf=np.array(z)
        a=self.relu(buf)
        self.l_a.append(a)
        return a
    
    def backprop(self):
        self.da_dz=[]
        self.dz_dw=[]
        self.dc_dw=[]
        self.dc_da=[]
        self.dc_db=[]
        
        #dc/dalast i|=i|-i|
        self.dc_dalast=-(self.y-self.a)
#         self.dalast_smax=self.a*(1.-self.a)
#         self.dc_dalast*=self.dalast_smax
        
        for i in range(len(self.l_a)-1):
            if(i==0):
                #da_dz i|b-=i|b-
                da_dz=self.l_a_buf[-1-i]
                da_dz[da_dz>=0]=1.0
                da_dz[da_dz<0]=self.alpha
                self.da_dz.append(da_dz)
                #dc_dz i|b-=i|b-*i|b-
                dc_dz=self.dc_dalast*da_dz
                #dc_db i|b-=i|b-*i|b-
                num_i=len(dc_dz[:,0])
                dc_db=(np.sum(dc_dz,axis=1)/self.batch).reshape(num_i,1)
                self.dc_db.append(dc_db)
                #dc_da j|b-=sum(i|*i|j-(notmatmul),axis=0).transpose()b-
                #num of pixels for one elem of batch
                num_i=len(dc_dz[:,0])
                num_j=len(self.w[-1-i][0,:])
                
                dc_da=np.asarray([np.sum(dc_dz[:,k].reshape(num_i,1)*self.w[-1-i],axis=0) for k in range(self.batch)]).reshape(self.batch,num_j).transpose()
                self.dc_da.append(dc_da)
                #dz_dw b|j-=j|b-
                dz_dw=self.l_a_buf[-2-i].transpose()
                self.dz_dw.append(dz_dw)
                #dc_dw(i|j-=i|*j-notmatmul)
                #dc_dw=(i|*j-)b-
                buf=[dc_dz[:,k].reshape(num_i,1)*dz_dw[k,:] for k in range(self.batch)]
                dc_dw=0
                for k in range(self.batch):
                    dc_dw+=buf[k]
                dc_dw=dc_dw/self.batch
                self.dc_dw.append(dc_dw)
            else:
                #da_dz i|b-=i|b-
                da_dz=self.l_a_buf[-1-i]
                da_dz[da_dz>=0]=1.0
                da_dz[da_dz<0]=self.alpha
                self.da_dz.append(da_dz)
                #dz_dw b|j-=j|b-(notmatmul)
                dz_dw=self.l_a_buf[-2-i].transpose()
                self.dz_dw.append(dz_dw)
                #dc_dw i|j-=i|*i|*j-(not matmul)
                num_i=len(dc_da[:,0])
                buf=[(dc_da[:,k]*da_dz[:,k]).reshape(num_i,1)*dz_dw[k,:] for k in range(self.batch)]
                dc_dw=0
                for k in range(self.batch):
                    dc_dw+=buf[k]
                dc_dw=dc_dw/self.batch
                self.dc_dw.append(dc_dw)
                #dc_dz i|b-=i|b-*i|b-
                dc_dz=dc_da*da_dz
                #dc_db i|b-=i|b-
                num_i=len(dc_dz[:,0])
                dc_db=(np.sum(dc_dz,axis=1)/self.batch).reshape(num_i,1)
                self.dc_db.append(dc_db)
                #dc_da j-=i|*i|*i|j-()
                num_j=len(self.w[-1-i][0,:])
                num_i=len(dc_dz[:,0])
                dc_da=np.asarray([np.sum(dc_dz[:,k].reshape(num_i,1)*self.w[-1-i],axis=0) for k in range(self.batch)]).reshape(self.batch,num_j).transpose()
                #dc_da j|=j-
                #dc_da=dc_da.transpose()
                self.dc_da.append(dc_da)
                
        #list->nparray
        self.dc_dw=np.array(self.dc_dw[::-1])
        self.dc_db=np.array(self.dc_db[::-1])

    def update(self):
        self.w-=self.eta*self.dc_dw
        self.b-=self.eta*self.dc_db

    def train(self):
        self.l_w=[]
        self.l_b=[]
        self.l_z=[]
        self.l_a=[]

        self.x,self.y=self.get_tx(self.batch)
        self.l_a.append(self.x)
        self.a=self.x

        for i in range(len(self.size)-1):
            self.a=self.forward(self.w[i],self.a,self.b[i])
#         self.a=self.softmax(self.a)
        self.lossfun()
        self.l_a_buf=copy.deepcopy(self.l_a)
        self.backprop()
        self.update()

    #leaky relu
    def relu(self,x):
        x[x<0]=self.alpha*x[x<0]
        return x

    def softmax(self,x):
        shiftx=x-np.max(x)
        exps=np.exp(shiftx)
        return exps/np.sum(exps)

    def lossfun(self):
        self.loss=(self.y-self.a)**2/2.
        self.t_loss=np.sum(self.loss,axis=0)


class Img(object):
    def __init__(self):
        self.im=[]
        self.imgpath=[]
        self.imgpath=[i for i in glob.glob(path)]
        self.imgname=map(os.path.basename,self.imgpath)
        self.d_index=[int(self.imgname[i][0]) for i in range(len(self.imgname))]
        
    def preprocess(self):
        #without normalization
        self.im=[Image.open(self.imgpath[i]).convert('L') for i in range(len(self.imgpath))]
        self.im=[np.array(self.im[i].resize((10,10))).reshape(100,1)/255. for i in range(len(self.imgpath))]
        
        
nn=Network([1,200,1])
start=time.time()

loss=[]
for p in range(100000):
    nn.train()
    loss.append(np.sum(nn.t_loss))
plt.subplot()
plt.plot(range(len(loss)),loss,marker='o',color='b')
print 'loss',loss
plt.show()
print 'time',time.time()-start
print 'time(min)',(time.time()-start)/60.

# for i in range(len(nn.w)):
#     w_name='w'+str(i)+'.txt'
#     b_name='b'+str(i)+'.txt'
#     np.savetxt(w_name,nn.w[i])
#     np.savetxt(b_name,nn.b[i])
