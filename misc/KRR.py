import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools
#from scipy.sparse import csr_matrix

import time




import os


X_train = pd.read_csv('data/Xtr0_mat100.csv', sep=' ', header=None)
X_test = pd.read_csv('data/Xte0_mat100.csv', sep=' ', header=None)
y_train = pd.read_csv('data/Ytr0.csv', index_col=0)

#Loss

def sigmoid(z):
    
    return 1./(1. + np.exp(-z))

def logistic_loss(z):
    
    return -np.log(sigmoid(z))

def logistic_grad(z):
    
    return -sigmoid(-z)

def logistic_hess(z):
    
    return sigmoid(z)*sigmoid(-z)



#Kernels

def linear_kernel(u,v):
    
    return np.dot(u,v)

def gaussian_kernel(x,y, sigma=1):
    
    return np.exp(-np.linalg.norm(x-y, ord=2)/(2*sigma**2))

def compute_gram_matrix(X, kernel):
    
    n = X.shape[0]
    gram_matrix = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i,n):
            gram_matrix[i, j] = kernel(X[i], X[j])
    
    gram_matrix = gram_matrix + gram_matrix.T + np.diag(gram_matrix.diagonal())    
    return gram_matrix


#Utilitaire

def compute_prediction(f_hat):
    
    return np.sign(f_hat)


def compute_margin(f_hat, y):
    
    return y*f_hat

def compute_functionnal_norm(K, alpha):
    """
    K -- Kernel matrix
    alpha -- coordinates
    
    """
    
    product = np.dot(K, alpha)
    
    return np.dot(alpha, product)
    
def normalize_data(data):
    
    mean_data = np.mean(data)
    std_data = np.std(data)
    
    data = (data - mean_data)/std_data
    
    return data
    
def preprocess(X_train, X_test, y_train, val_split = False):
    X_train = X_train.to_numpy()
    y_train = (2*y_train - 1).to_numpy().T[0] # y=-1,1
    if val_split == True:
        X_train, X_val = np.split(X_train, [1500])
        y_train, y_val = np.split(y_train, [1500])
    X_test = X_test.to_numpy()    
    #rescaling
    mean_train = X_train.mean()
    std_train = X_train.std()
    X_train = (X_train - mean_train)/std_train
    if val_split == True:
        X_val = (X_val - mean_train)/std_train
    X_test = (X_test - mean_train)/std_train
    if val_split == True :
        return X_train, X_test, X_val, y_train, y_val
    else:
        return X_train, X_test, y_train

X_train, X_test, X_val, y_train, y_val = preprocess(X_train, X_test, y_train, True)

class KRR():
    
    def __init__(self, kernel="gauss", sigma=2):
        self.sigma = sigma
        if kernel == "gauss":
            self.K = self.gaussian_kernel
        
        if kernel =="linear":
            self.K = self.linear_kernel
    
    def fit(self,X, y, C=1e0, gram_normalization=False):
        self.C = C
        self.X = X
        self.y = y.astype(np.double)
        self.n = self.X.shape[0]
        
        self.K_train = self.compute_gram_matrix(self.X)
        
        if gram_normalization:
            self.K_train = self.normalize_gram_matrix(self.K_train)
            
            
        proxy = np.linalg.inv(self.K_train + self.n*self.C*np.identity(self.n))
        
        self.alpha = np.dot(proxy, self.y)
        
        pass
    
    def predict_one(self,x):
        prods = np.array([self.K(x,X) for X in self.X])
        return np.sign(np.dot(prods, self.alpha))
        pass
    
    def predict(self,X):
        return np.array([self.predict_one(x) for x in X]).T
    
    def score(self,X,y):
        return np.mean(self.predict(X) == y)
    
    def gaussian_kernel(self,x,y):
        return np.exp(-np.linalg.norm(x-y, ord=2)/(2*(self.sigma)**2))
    
    def linear_kernel(self, x, y):
        
        return np.dot(x,y)
    
    def compute_gram_matrix(self,X):   
        gram_matrix = np.zeros((self.n,self.n))

        for i in range(self.n):
            for j in range(i,self.n):
                gram_matrix[i, j] = self.K(X[i], X[j])

        gram_matrix = gram_matrix + gram_matrix.T - np.diag(gram_matrix.diagonal())
        
        return gram_matrix

for i, epsilon in enumerate([1e-3, 1e-2, 1e-1, 1e0, 1e1]):
    
    print('Epsilon: {}'.format(epsilon))
    clf0 = KRR(sigma=1.5)
    clf0.fit(X_train, y_train, C=epsilon)
    print('Training score:')
    print(clf0.score(X_train,y_train))
    print('Validation score:')
    print(clf0.score(X_val,y_val))