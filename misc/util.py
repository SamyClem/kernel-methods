import numpy as np 
import itertools 


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

def compute_dictionary(k, alphabet):
        """
        alphabet -- 'ATGC' dans notre cas
        k -- longueur des mots de notre dictionnaire
        
        
        output -- liste de tout les mots de taille k de notre alphabet
        """
        
        return np.array([''.join(i) for i in itertools.product(alphabet, repeat = k)])


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
    
def compute_gram_matrix(X, kernel): 

        n = X.shape[0]
        gram_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i,n):
                gram_matrix[i, j] = kernel(X[i], X[j])

        gram_matrix = gram_matrix + gram_matrix.T - np.diag(gram_matrix.diagonal())
        
        return gram_matrix

def sigmoid(z):
    
    return 1./(1. + np.exp(-z))

def logistic_loss(z):
    
    return -np.log(sigmoid(z))

def logistic_grad(z):
    
    return -sigmoid(-z)

def logistic_hess(z):
    
    return sigmoid(z)*sigmoid(-z)