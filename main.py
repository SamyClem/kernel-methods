import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from cvxopt import matrix, solvers
import os

X_train0 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xtr0.csv',  sep=',', header=0, index_col=0)
X_test0 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xte0.csv', sep=',', header=0, index_col=0)
y_train0 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Ytr0.csv', index_col=0)
X_train1 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xtr1.csv', sep=',', header=0, index_col=0)
X_test1 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xte1.csv', sep=',', header=0, index_col=0)
y_train1 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Ytr1.csv', index_col=0)
X_train2 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xtr2.csv', sep=',', header=0, index_col=0)
X_test2 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Xte2.csv', sep=',', header=0, index_col=0)
y_train2 = pd.read_csv('/kaggle/input/machine-learning-with-kernel-methods-2021/Ytr2.csv', index_col=0)

def convert_to_array(X_train, X_test, y_train):
    
    X_train = X_train['seq'].values
    X_test = X_test['seq'].values
    y_train = (2*y_train - 1).to_numpy().T[0] # y=-1,1

    return X_train, X_test, y_train

def convert_to_array_val(X_train, X_test, y_train):
    X_train, X_val = np.split(X_train, [1500])
    y_train, y_val = np.split(y_train, [1500])
    
    X_train = X_train['seq'].values
    X_val = X_val['seq'].values
    X_test = X_test['seq'].values
    y_train = (2*y_train - 1).to_numpy().T[0] # y=-1,1
    y_val = (2*y_val - 1).to_numpy().T[0]

    return X_train, X_test, X_val, y_train, y_val

import itertools 


def compute_dictionary(k, alphabet):
    """
    alphabet -- 'ATGC' dans notre cas
    k -- longueur des mots de notre dictionnaire
    
    
    output -- liste de tout les mots de taille k de notre alphabet
    """
    
    return np.array([''.join(i) for i in itertools.product(alphabet, repeat = k)])


#Je calcule les features et pas directement le kernel pour + de souplesse, 
#vu que le kernel sobtient juste en faisant un ps

def k_spectrum_features(k, X, dictionary):
    
    #Dimension of the pb
    n = X.shape[0]
    p = dictionary.shape[0]
    
    #Initialisation of feature matrix
    X_feature = np.zeros((X.shape[0], dictionary.shape[0]))
        
    
    for i, word in enumerate(X):
        l = len(word)
        k_gram = np.array([''.join(word[u:u+k]) for u in range(l-k+1)])
        
        for j, key in enumerate(dictionary):
            
            for w in k_gram:
                if w == key:
                    X_feature[i, j] += 1
        
    
    return X_feature
  
X_train0, X_test0, y_train0 = convert_to_array(X_train0, X_test0, y_train0)
X_train1, X_test1, y_train1 = convert_to_array(X_train1, X_test1, y_train1)
X_train2, X_test2, y_train2 = convert_to_array(X_train2, X_test2, y_train2)

alphabet = 'ATGC'

k = 4
dictionary = compute_dictionary(k, alphabet)

X_train0_features = k_spectrum_features(k, X_train0, dictionary)
spectrum_kernel_train0 = np.dot(X_train0_features, X_train0_features.T)
X_test0_features = k_spectrum_features(k, X_test0, dictionary)
#X_val0_features = k_spectrum_features(k, X_val0, dictionary)

k=5
dictionary = compute_dictionary(k, alphabet)

X_train1_features = k_spectrum_features(k, X_train1, dictionary)
spectrum_kernel_train1 = np.dot(X_train1_features, X_train1_features.T)
X_test1_features = k_spectrum_features(k, X_test1, dictionary)
#X_val1_features = k_spectrum_features(k, X_val1, dictionary)
X_train2_features = k_spectrum_features(k, X_train2, dictionary)
spectrum_kernel_train2 = np.dot(X_train2_features, X_train2_features.T)
X_test2_features = k_spectrum_features(k, X_test2, dictionary)
#X_val2_features = k_spectrum_features(k, X_val2, dictionary)

#############################################

class SVM_K():
    def __init__(self, C=1e0):
        self.C = C
    
    def fit(self, K_train, X_train_features, y):
        self.X_train_features = X_train_features
        self.K_train = K_train
        self.y = y.astype(np.double)
        self.n = self.K_train.shape[0]
        
        P = matrix(self.K_train)
        q = -matrix(self.y)
        G = matrix(np.vstack([np.diag(self.y),np.diag(-self.y)]))
        h = matrix(np.hstack([self.C*np.ones(self.n),np.zeros(self.n)]))
        
        self.sol = solvers.qp(P,q,G,h)       
        pass
    
    def predict_one(self,x_features):
        phi = np.dot(self.X_train_features, x_features)
        return np.sign(np.dot(np.array(self.sol['x']).T[0], phi))
        pass
    
    def predict(self,X_features):
        return np.array([self.predict_one(x) for x in X_features])
    
    def score(self,X,y):
        return np.mean(self.predict(X) == y)
#########################################

C=0.1
clf0 = SVM_K(C)
C=0.01
clf1 = SVM_K(C)
clf2 = SVM_K(C)

clf0.fit(spectrum_kernel_train0, X_train0_features, y_train0)
clf1.fit(spectrum_kernel_train1, X_train1_features, y_train1)
clf2.fit(spectrum_kernel_train2, X_train2_features, y_train2)

predict0 = clf0.predict(X_test0_features)
predict1 = clf1.predict(X_test1_features)
predict2 = clf2.predict(X_test2_features)

predictions = pd.DataFrame({'Id': range(3000), 'Bound' : ((np.hstack([predict0, predict1, predict2]) + 1)/2).astype('int')}).set_index('Id')
predictions.to_csv('submission_spectrum.csv')

print("predictions saved as submission_spectrum.csv")
