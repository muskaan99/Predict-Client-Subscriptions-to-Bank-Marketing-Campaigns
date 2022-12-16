
import pandas as pd
import numpy as np
import copy
import random
import sys
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
import seaborn as sns
import matplotlib.pyplot as plt

def performance_metrics(y,y_pred):
    '''
    Function to find Sensitivity, Specificity and Accuracy given the target label and predicted label
    
    Parameters:
    y- target label
    y_pred- predicted label

    Return:
    Specificity,Sensitivity,Accuracy scores
    '''
    result_lst=[]
    conf=confusion_matrix(y, y_pred, labels = [0, 1])
    colormap = sns.color_palette("Pastel1")
    plt.figure(figsize=(6,6))  
    sns.heatmap(conf, annot=True, fmt='g', cmap=colormap)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    tn, fp, fn, tp = conf[0][0],conf[0][1],conf[1][0],conf[1][1]
    specificity = tn / (tn+fp)
    sensitivity=tp/(tp+fn)
    acc=accuracy_score(y,y_pred)
    return [specificity,sensitivity,acc]


def test_model(best_model,X_test,y_test,threshold):
    '''
    Function to find the predicted labels on test set given a trained model
    
    Parameters:
    best_model- target label
    X_test,y_test- predicted label
    threshold- cut off threshold for probability

    Return:
    List of performance metrics Specificity,Sensitivity,Accuracy scores
    '''
    y_hat=best_model.predict_proba(X_test)
    y_pred = np.zeros(len(y_hat))
    for i in range(len(y_hat)):
      if y_hat[i, 1] > threshold:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
        y_pred = y_pred.astype(int)
    return performance_metrics(y_test,y_pred)

###########################################################################################################################
# EM Algorithm Functions
def learn_params(X_l, y_l):
    
    
    N = X_l.shape[0]
    phi = X_l[y_l == 1].shape[0] / N
    mu0 = np.sum(X_l[y_l == 0], axis=0) / X_l[y_l == 0].shape[0]
    mu1 = np.sum(X_l[y_l == 1], axis=0) / X_l[y_l == 1].shape[0]
    sigma0 = np.cov(X_l[y_l == 0].T, bias= True)
    sigma1 = np.cov(X_l[y_l == 1].T, bias=True)
    
    return {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}


def estep(X_u, params):
    ''''''
    
    np.log([multivariate_normal(params["mu0"], params["sigma0"],allow_singular=True).pdf(X_u), 
            multivariate_normal(params["mu1"], params["sigma1"],allow_singular=True).pdf(X_u)])
    
    log_likely = np.log([1-params["phi"], params["phi"]])[np.newaxis, ...] + np.log([multivariate_normal(params["mu0"], params["sigma0"],allow_singular=True).pdf(X_u),
            multivariate_normal(params["mu1"], params["sigma1"],allow_singular=True).pdf(X_u)]).T
    
    log_likely_norm = logsumexp(log_likely, axis=1)
    
    return log_likely_norm, np.exp(log_likely - log_likely_norm[..., np.newaxis])

def m_step(X_u, params):
    ''''''
    
    len = X_u.shape[0]
    _, statistics = estep(X_u, params)
    statistic0 = statistics[:, 0]
    statistic1 = statistics[:, 1]
    statistic1_sum = np.sum(statistic1)
    statistic0_sum = np.sum(statistic0)
    
    phi = (statistic1_sum/len)
    
    mu0 = (statistic0[..., np.newaxis].T.dot(X_u)/statistic0_sum).flatten()
    mu1 = (statistic1[..., np.newaxis].T.dot(X_u)/statistic1_sum).flatten()
    diff_0 = X_u - mu0
    sigma0 = diff_0.T.dot(diff_0 * statistic0[..., np.newaxis]) / statistic0_sum
    
    diff_1 = X_u - mu1
    sigma1 = diff_1.T.dot(diff_1 * statistic0[..., np.newaxis]) / statistic1_sum
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    
    return params


def get_avg_likelihood(X_u, params):
    ''''''
    
    likelihood, _ = estep(X_u, params)
    return np.mean(likelihood)

def run_em(X_u, params, max_iter):
    ''''''
    
    avg_loglikelihood_val = []
    count = 0

    while count < max_iter:
        avg_loglikelihood = get_avg_likelihood(X_u, params)
        avg_loglikelihood_val.append(avg_loglikelihood)
        params = m_step(X_u, params)
        count+=1
  
    
    _, final_log_likely = estep(X_u, params)
    prediction = np.argmax(final_log_likely, axis = 1)
    
    return prediction, avg_loglikelihood_val

def em(X_l, y_l, X_u):
    ''''''
    
    max_iter = 51
    learned_params = learn_params(X_l, y_l)
    labels, avg_likely = run_em(X_u, learned_params, max_iter)
    # unique, counts = np.unique(labels, return_counts=True)
    
    return labels

def em_ssl(X_l, y_l, X_u,y_u):
  ''''''

  labels=em(X_l, y_l, X_u)
  X_l = np.concatenate((X_l, X_u), axis = 0)
  y_l = np.concatenate((y_l, labels), axis = 0)

  params = learn_params(X_l, y_l)
  weights = [1 - params["phi"], params["phi"]]
  means = [params["mu0"], params["mu1"]]
  covariances = [params["sigma0"], params["sigma1"]]

  clf = GaussianMixture(n_components = 2,
                        covariance_type='full',
                        tol=0.01,
                        max_iter=11,
                        weights_init=weights,
                        means_init=means,
                        precisions_init=covariances)

  clf.fit(X_l, y_l)
  return clf
###########################################################################################################################


# SSL:Prop 1 NN
class Prop_1NN:
    
    def __init__(self,X_l, y_l, X_u): 
        self.X_l =X_l
        self.y_l =y_l
        self.X_u =X_u
       
    def train_prop_1nn(self,name,threshold_cutoff,final_threshold):
        clf = KNeighborsClassifier(n_neighbors = 17)
        label_count = len(self.X_l)
        unlabel_count = len(self.X_u)
        y_pred = np.zeros(unlabel_count)
        while unlabel_count > 0:
            shift_count = 0
            clf.fit(self.X_l, self.y_l)
            y_hat = clf.predict_proba(self.X_u)
            X_u_copy = copy.deepcopy(self.X_u)
            index = []
            thresh =threshold_cutoff

            for i in range(len(y_hat)):
                if y_hat[i][0] > thresh or y_hat[i][1] > thresh : 
                   
                    self.X_l = np.append(self.X_l, [X_u_copy[i]], axis = 0)
                    self.y_l = np.append(self.y_l, np.argmax(y_hat[i]))
                    index.append(i)
                    shift_count+=1
                
            self.X_u = np.delete(self.X_u, index, axis = 0)
            if shift_count == 0:
                break

            label_count+=shift_count
            unlabel_count-=shift_count
    
        clf.fit(self.X_l, self.y_l)

        y_hat=clf.predict_proba(self.X_l)
        y_pred = np.zeros(len(y_hat))
        for i in range(len(y_hat)):
            if y_hat[i, 1] > final_threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
                y_pred = y_pred.astype(int)
        values=performance_metrics(self.y_l,y_pred)
        result_train=[name]+values
        return clf,result_train

    def test_prop_1nn(self,name,model,X_test_final,y_test,threshold_value):
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test


###########################################################################################################################

# SSL: Expectation Maximization
class Expectation_Maximization:
    
    def __init__(self,X_l, y_l, X_u,y_u): 
        self.X_l =X_l
        self.y_l =y_l
        self.X_u =X_u
        self.y_u =y_u
    
    def train_em(self,name,X_train,y_train,threshold):
        best_em_model=em_ssl(self.X_l, self.y_l,self.X_u,self.y_u)
        
        values= test_model(best_em_model,X_train,y_train,threshold)
        result_train=[name]+values
        return best_em_model,result_train

    def test_em(self,name,model,X_test_final,y_test,threshold_value):
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test
###########################################################################################################################

# SSL: Label Propagation 
class Label_Propagation:

    def __init__(self,X_l, y_l, X_u,y_u): 
        self.X_l =X_l
        self.y_l =y_l
        self.X_u =X_u
        self.y_u =y_u

    def train_label_prop(self,name,threshold):
        clf = LabelPropagation()
        X_merge = np.concatenate((self.X_l, self.X_u), axis = 0)
        y_unlab = np.zeros(len(self.X_u))
        y_unlab[y_unlab == 0] = -1
        y_merge = np.concatenate((self.y_l, self.y_u), axis = 0)
        clf.fit(X_merge, y_merge)
        y_hat = clf.predict_proba(X_merge)
        y_pred = np.zeros(len(y_hat))

        for i in range(len(y_hat)):
            if y_hat[i, 1] > threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        y_pred = y_pred.astype(int)
        values= performance_metrics(y_merge,y_pred)
        result_train=[name]+values
        return clf,result_train

    def test_label_prop(self,name,model,X_test_final,y_test,threshold_value):
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test  

###########################################################################################################################
# ### EXPERIMENT

# SSL: SSL Logistic Regression
class SSL_Log_Reg:
    
    def __init__(self,X_l, y_l, X_u,y_u): 
        self.X_l =X_l
        self.y_l =y_l
        self.X_u =X_u
        self.y_u =y_u
    
    def train_ssl_log_reg(self,name,threshold_cutoff,final_threshold):
        clf = LogisticRegression(C = 0.1, solver = 'newton-cg')
        label_count = len(self.X_l)
        unlabel_count = len(self.X_u)
        y_pred = np.zeros(unlabel_count)
        while unlabel_count > 0:
            shift_count = 0
            clf.fit(self.X_l, self.y_l)
            y_hat = clf.predict_proba(self.X_u)
            
            X_u_copy = copy.deepcopy(self.X_u)
            index = []
            thresh =threshold_cutoff
            for i in range(len(y_hat)):
                if y_hat[i][0] > thresh or y_hat[i][1] > thresh : 
                    self.X_l = np.append(self.X_l, [X_u_copy[i]], axis = 0)
                    self.y_l = np.append(self.y_l, np.argmax(y_hat[i]))
                    index.append(i)
                    shift_count+=1
                
            self.X_u = np.delete(self.X_u, index, axis = 0)
            if shift_count == 0:
                break

            label_count+=shift_count
            unlabel_count-=shift_count
    
        clf.fit(self.X_l, self.y_l)

        y_hat=clf.predict_proba(self.X_l)
        y_pred = np.zeros(len(y_hat))
        for i in range(len(y_hat)):
            if y_hat[i, 1] > final_threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
                y_pred = y_pred.astype(int)
        values=performance_metrics(self.y_l,y_pred)
        result_train=[name]+values
        return clf,result_train

    def test_ssl_log_reg(self,name,model,X_test_final,y_test,threshold_value):
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test



        
###########################################################################################################################
###########################################################################################################################