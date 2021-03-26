import numpy as np
import pandas as pd
from util import *


S = np.identity(4)
alphabet_dict = dict({"A": 0, "T":1, "C":2, "G":3})


def penalty_gap(n, d, e):
    if n==0:
        return 0
    else:
        return d + e*(n-1)
    
class LAKernel():

    def __init__(self, S, d, e, beta):
        self.S = S
        self.d = d
        self.e = e
        self.beta = beta
        

    def la_kernel(x,y):
        
        u = len(x)
        v = len(y)
        
        mat_M = np.zeros((u, v))
        mat_X = np.zeros((u, v))
        mat_Y = np.zeros((u, v))
        mat_X2 = np.zeros((u, v))
        mat_Y2 = np.zeros((u, v))
        
        for i in range(1, u):
            for j in range(1, v):
            
                #Compute index of substitution matrix
                x_i = alphabet_dict[x[i]]
                y_j = alphabet_dict[y[j]]

                mat_M[i, j] = (1 + mat_X[i-1, j-1] + mat_Y[i-1, j-1] + mat_M[i-1, j-1])* np.exp(self.beta*self.S[x_i, y_j])
                
                mat_X[i, j] = np.exp(self.beta*self.d)*mat_M[i-1, j] + np.exp(self.beta*self.e)*mat_X[i-1, j]
                mat_Y[i, j] = np.exp(self.beta*self.d)*(mat_M[i, j-1] + mat_X[i, j-1]) + np.exp(self.beta*self.e)*mat_Y[i, j-1]
                
                mat_X2[i, j] = mat_M[i-1, j] + mat_X2[i-1, j]
                mat_Y2[i, j] = mat_M[i, j-1] + mat_X2[i, j-1] + mat_Y2[i, j-1]  
        
        
        
        return 1 + mat_X2[u-1, v-1] + mat_Y2[u-1, v-1] + mat_M[u-1, v-1]


    

if __name__ == '__main__':


    print("Loading data\n")
    X_train0 = pd.read_csv('data/Xtr0.csv',  sep=',', header=0, index_col=0)
    X_test0 = pd.read_csv('data/Xte0.csv', sep=',', header=0, index_col=0)
    y_train0 = pd.read_csv('data/Ytr0.csv', index_col=0)
    X_train1 = pd.read_csv('data/Xtr1.csv', sep=',', header=0, index_col=0)
    X_test1 = pd.read_csv('data/Xte1.csv', sep=',', header=0, index_col=0)
    y_train1 = pd.read_csv('data/Ytr1.csv', index_col=0)
    X_train2 = pd.read_csv('data/Xtr2.csv', sep=',', header=0, index_col=0)
    X_test2 = pd.read_csv('data/Xte2.csv', sep=',', header=0, index_col=0)
    y_train2 = pd.read_csv('data/Ytr2.csv', index_col=0)

    S = np.identity(4)
    alphabet_dict = dict({"A": 0, "T":1, "C":2, "G":3})

    
    lak = LAKernel(S, d=1, e=1, beta=0.5)
    
    la_kernel_train0 = compute_gram_matrix(X_train0, lak.la_kernel)
    la_kernel_train1 = compute_gram_matrix(X_train1, lak.la_kernel)
    la_kernel_train1 = compute_gram_matrix(X_train2, lak.la_kernel)

    la_kernel_test0 = compute_gram_matrix(X_test0, lak.la_kernel)
    la_kernel_test1 = compute_gram_matrix(X_test1, lak.la_kernel)
    la_kernel_test1 = compute_gram_matrix(X_test2, lak.la_kernel)




    


