import numpy as np
from util import *
import pandas as pd 
import os



class WKRR():
    
    def __init__(self, kernel="gauss", C, sigma=2):
        
        self.C = C
        self.sigma = sigma
        if kernel == "gauss":
            self.K = self.gaussian_kernel
        
        if kernel =="linear":
            self.K = self.linear_kernel
    
    def fit(self,X, y, w, gram_normalization=False):
        
        self.X = X
        self.y = y.astype(np.double)
        self.n = self.X.shape[0]
        
        self.K_train = self.compute_gram_matrix(self.X)
        
        if gram_normalization:
            self.K_train = self.normalize_gram_matrix(self.K_train)
        
        
        sqrt_w = np.sqrt(w)
        W = np.diag(w)
        sqrt_W = np.diag(sqrt_w)

        weighted_K = np.dot(np.dot(sqrt_W, self.K_train), sqrt_W)

        proxy = np.linalg.inv(weighted_K + self.n*self.C*np.identity(self.n))
        weighted_proxy = np.dot(np.dot(sqrt_W, proxy), sqrt_W)
        
        self.alpha = np.dot(weighted_proxy, self.y)      
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




class KernelLogisticRegression():
    
    def __init__(self, kernel="gauss", C, sigma=2):
        self.C = C
        self.sigma = sigma
        if kernel == "gauss":
            self.K = self.gaussian_kernel
        
        if kernel =="linear":
            self.K = self.linear_kernel
    
    def fit(self,X, y, n_iter=100, gram_normalization=False):
        
        self.X = X
        self.y = y.astype(np.double)
        self.n = self.X.shape[0]
        
        self.K_train = self.compute_gram_matrix(self.X)
        
        if gram_normalization:
            self.K_train = self.normalize_gram_matrix(self.K_train)
        
        

        #Initialisation

        self.alpha = np.ones(self.n)
        
        m_0 = np.dot(self.K_train, self.alpha)
        
        P_0 = logistic_grad(self.y*m_0)
        W_0 = logistic_hess(self.y*m_0)
        
        z_0 = m_0 + self.y/sigmoid(self.y*m_0)
        
        wkrr = WKRR(C=self.C)

        for i in range(n_iter):
            
            wkrr.fit(self.X, z_0, W_0)
            self.alpha = wkrr.alpha

            m_0 = np.dot(self.K_train, self.alpha)
            P_0 = logistic_grad(self.y*m_0)
            W_0 = logistic_hess(self.y*m_0)
            z_0 = m_0 + self.y/sigmoid(self.y*m_0)
                   
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


if __name__ =='__main__':

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


    print("Cleaning data\n")  
    X_train0, X_test0, y_train0 = convert_to_array(X_train0, X_test0, y_train0)
    X_train1, X_test1, y_train1 = convert_to_array(X_train1, X_test1, y_train1)
    X_train2, X_test2, y_train2 = convert_to_array(X_train2, X_test2, y_train2)

    C=0.1
    clf0 = KernelLogisticRegression(C)
    C=0.01
    clf1 = KernelLogisticRegression(C)
    clf2 = KernelLogisticRegression(C)


    print("Fitting classifiers\n")
    clf0.fit(X_train0, y_train0)
    clf1.fit(X_train1, y_train1)
    clf2.fit(X_train2, y_train2)

    print("Computing predictions on the test set\n")
    predict0 = clf0.predict(X_test0)
    predict1 = clf1.predict(X_test1)
    predict2 = clf2.predict(X_test2)

    predictions = pd.DataFrame({'Id': range(3000), 'Bound' : ((np.hstack([predict0, predict1, predict2]) + 1)/2).astype('int')}).set_index('Id')
    predictions.to_csv('results/submission.csv')

    print("Predictions saved as results/submission.csv\n")