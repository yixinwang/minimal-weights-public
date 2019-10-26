import numpy as np 
import numpy.random as npr
import math
import pandas as pd


def WongChanSimCov(n):
    Z = npr.normal(size=(n, 10))
    X = np.zeros((n, 10))
    X[:,0] = np.exp(Z[:,0]/2.)
    X[:,1] = Z[:,1]/(1+np.exp(Z[:,0]))
    X[:,2] = (Z[:,0]*Z[:,2]/25.+0.6)**3
    X[:,3] = (Z[:,1]+Z[:,3]+20)**2
    X[:,4:] = Z[:,4:]
    return n, Z, X

def WongChanSimPS(n, Z, X):
    p = np.exp(-Z[:,1]-0.1*Z[:,4]) / (1.+np.exp(-Z[:,1]-0.1*Z[:,4]))
    T = npr.binomial(1, p)
    return p, T

def WongChanSimOutA(n, Z, X, T):
    Y = 210 + \
        (1.5*T-0.5) * (27.4*Z[:,1]+13.7*Z[:,2]+13.7*Z[:,3]+13.7*Z[:,4]) + \
        npr.normal(size=n)
    Y1 = 210 + \
        (1.5*1-0.5) * (27.4*Z[:,1]+13.7*Z[:,2]+13.7*Z[:,3]+13.7*Z[:,4]) + \
        npr.normal(size=n)
    Y0 = 210 + \
        (1.5*0-0.5) * (27.4*Z[:,1]+13.7*Z[:,2]+13.7*Z[:,3]+13.7*Z[:,4]) + \
        npr.normal(size=n)
    return Y, Y1, Y0

def WongChanSimOutB(n, Z, X, T):
    Y = Z[:,1]*(Z[:,2]**3)*(Z[:,3]**2)*Z[:,4] + Z[:,4]*(np.abs(Z[:,1]))**0.5 + \
        npr.normal(size=n)
    Y1 = Z[:,1]*(Z[:,2]**3)*(Z[:,3]**2)*Z[:,4] + Z[:,4]*(np.abs(Z[:,1]))**0.5 + \
        npr.normal(size=n)
    Y0 = Z[:,1]*(Z[:,2]**3)*(Z[:,3]**2)*Z[:,4] + Z[:,4]*(np.abs(Z[:,1]))**0.5 + \
        npr.normal(size=n)
    return Y, Y1, Y0

def WongChanSimA(n=200):
    n, Z, X = WongChanSimCov(n)
    p, T = WongChanSimPS(n, Z, X)
    Y, Y1, Y0 = WongChanSimOutA(n, Z, X, T)
    return n, Z, X, p, T, Y, Y1, Y0

def WongChanSimB(n=200):
    n, Z, X = WongChanSimCov(n)
    p, T = WongChanSimPS(n, Z, X)
    Y, Y1, Y0 = WongChanSimOutB(n, Z, X, T)
    return n, Z, X, p, T, Y, Y1, Y0


if __name__ == '__main__':
    N = 100
    datdir = 'sim_datasets/'
    for i in range(N):
        n, Z, X, p, T, Y, Y1, Y0 = WongChanSimA(n=5000)
        simA = np.column_stack([Z, p, X, T, Y, Y1, Y0])
        np.savetxt(datdir+str(i)+'WongChanSimA.csv', simA, delimiter=',')
        n, Z, X, p, T, Y, Y1, Y0 = WongChanSimB(n=5000)
        simB = np.column_stack([Z, p, X, T, Y, Y1, Y0])
        np.savetxt(datdir+str(i)+'WongChanSimB.csv', simB, delimiter=',')



