# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:14:42 2018

@author: User
"""
import statistics
import numpy as np
import math
from numpy.linalg import inv

import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
from sklearn import datasets
def bubbleSort(w,v):
    n = len(w)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if w[j] < w[j+1] :
                w[j], w[j+1] = w[j+1],w[j]
                v[:,[j,j+1]] = v[:,[j+1,j]]
    return w,v
def readfile(filename):
    f=open('data.txt','r')
    data=f.read()
    data=data.split('\n')
    data=data[:len(data)-1]
    for i in range(len(data)):
        data[i]=[float(x) for x in data[i].split()]
    return data
def getmin(data):
    means=[statistics.mean(x) for x in zip(*data)]
    return means
def subcovariance(i,j,data,means):
    sums=0
    for ii in range(len(data)):
        sums+=(data[ii][i]-means[i])*(data[ii][j]-means[j])
    sums/=len(data)
    return sums
def getcovariance(data,means):
    covariance=[]
    for i in range(len(data[0])):
        g=[]
        for j in range(len(data[0])):
            g.append(subcovariance(i,j,data,means))
        covariance.append(g)
    return covariance
def getprobabilitydistribution(data,miu,omega,w):
    probabilitymatrix=[[0 for i in range(len(w))] for j in range(len(data))]
    for i in range(len(data)):
        summ=0.0
        for j in range(len(w)):
            probabilitymatrix[i][j]=w[j]*probabilityofgettingxifromkthdistribution(data,miu,omega,i,j)
            summ+=probabilitymatrix[i][j]
        probabilitymatrix[i]=[kk/summ for kk in probabilitymatrix[i]]
    return probabilitymatrix
def getparameterestimation(probabilitymatrix,data,koytacomponent):
    pp=[sum(element) for element in zip(*probabilitymatrix)]
    miu=np.array([[0.0 for i in range(len(data[0]))] for j in range(koytacomponent)])
    for i in range(koytacomponent):
        sums=0
        for j in range(len(data)):
            miu[i]+=probabilitymatrix[j][i]*np.array(data[j])
            sums+=probabilitymatrix[j][i]
        miu[i]/=sums
    
    omega=np.array([[[0.0 for i in range(len(data[0]))] for j in range(len(data[0]))] for k in range(koytacomponent)])
    
    for i in range(koytacomponent):
        sums=0
        for j in range(len(data)):
            #print('j='+str(j))
            A=np.array(data[j])-miu[i]
            #print(A)
            temp=[[0.0 for i in range(2)] for j in range(2)]
            for ii in range(2):
                for jj in range(2):
                    temp[ii][jj]=A[ii]*A[jj]
            temp=probabilitymatrix[j][i]*np.array(temp)
            omega[i]+=temp
            #print(omega[i])
            sums+=probabilitymatrix[j][i]
        print('str'+str(i))
        print(omega[i])
        omega[i]/=pp[i]
        print(omega[i])
    #print(omega)
    
    w=np.array([0.0 for i in range(koytacomponent)])
    w=[(element/len(data)) for element in pp]
    return miu,omega,w
def loglikelyhoodvalue(data,miu,omega,w):
    loglikely=0
    for i in range(len(data)):
        sums=0
        for j in range(len(w)):
            sums+=w[j]*probabilityofgettingxifromkthdistribution(data,miu,omega,i,j)
        sums=math.log(sums)
        loglikely+=sums
    return loglikely
            
    
def probabilityofgettingxifromkthdistribution(data,miu,omega,i,k):
    A=np.subtract(np.array(data[i]),np.array(miu[k]))
    A=np.transpose(A)
    
    B=inv(omega[k])
    
    C=np.subtract(np.array(data[i]),np.array(miu[k]))
    
    D=np.matmul(A,B)
    
    D=np.matmul(D,np.array(C))
    
    D=-0.5*D
    
    E=math.pow((2*math.pi),len(data[i])/2)
    
    F=abs(np.linalg.det(np.array(omega[k])))
    F=math.pow(F,0.5)
    E=(1/(E*F))*math.exp(D)
    
    return E
    
def init(dimension,component,data):
    miu=[[np.random.uniform()*10 for jj in range(dimension)] for i in range(component)]
    omega=np.array([[[0.0 for i in range(len(data[0]))] for j in range(len(data[0]))] for k in range(component)])
    
    for i in range(component):
        sums=0
        for j in range(len(data)):
            #print('j='+str(j))
            A=np.array(data[j])-miu[i]
            #print(A)
            temp=[[0.0 for i in range(2)] for j in range(2)]
            for ii in range(2):
                for jj in range(2):
                    temp[ii][jj]=A[ii]*A[jj]
            temp=probabilitymatrix[j][i]*np.array(temp)
            omega[i]+=temp
            #print(omega[i])
            sums+=probabilitymatrix[j][i]
        print('str'+str(i))
        print(omega[i])
        omega[i]/=1/component
        print(omega[i])
    w=[np.random.uniform() for i in range(component)]
    summ=sum(w)
    w=[i/summ for i in w]
    return miu,omega,w
        
if __name__=="__main__":
    
    data=readfile('data.txt')
    means=getmin(data)
    covariance=getcovariance(data,means)
    print(len(covariance))
    print(len(covariance[0]))
    #print(covariance[0][0])
    w, v = np.linalg.eig(covariance)
    w,v=bubbleSort(w,v)
    x=[]
    y=[]
    newdata=[]
    for i in range(len(data)):
        x.append(v[:,0].dot(data[i]))
        y.append(v[:,1].dot(data[i]))
        newdata.append([x[i],y[i]])
    plt.scatter(x,y)
    
    data=newdata
    miu,omega,w=init(2,3,data)
    i=0
    #print(omega)
    kk=inv(omega[0])
    #print(kk)
    probabilitymatrix=getprobabilitydistribution(data,miu,omega,w)
    #print(probabilitymatrix[0])
    #print(sum(probabilitymatrix[1]))
    miu,omega,w=getparameterestimation(probabilitymatrix,data,3)
    
    print(omega)
    
    while(1):
        print(i)
        L=loglikelyhoodvalue(data,miu,omega,w)
        print(L)
        probabilitymatrix=getprobabilitydistribution(data,miu,omega,w)
        miu,omega,w=getparameterestimation(probabilitymatrix,data,3)
        i+=1
        if(i==100):
            break
    plt.scatter(miu[:,[0]],miu[:,[1]])
    print('miu')
    print(miu)
    
    
    