import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,recall_score,accuracy_score,confusion_matrix
import seaborn as sns
from sklearn.neighbors import NearestCentroid
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
    
    conf=confusion_matrix(y, y_pred, labels = [0, 1])
    colormap = sns.color_palette("Greens")
    plt.figure(figsize=(6,6))  
    sns.heatmap(conf, annot=True, fmt='g', cmap=colormap)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    tn, fp, fn, tp = conf[0][0],conf[0][1],conf[1][0],conf[1][1]
    specificity = (tn / (tn+fp)).round(2)
    sensitivity=(tp/(tp+fn)).round(2)
    acc=(accuracy_score(y,y_pred)).round(2)
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
    

def supervised_training_model(classifier,parameter_dict,X,y,threshold):
    '''
    Function to perform grid search and compare performances by training a classifier 
    
    Parameters:
    best_model- target label
    X_test,y_test- predicted label
    threshold- cut off threshold for probability

    Return:
    List of performance metrics: Specificity,Sensitivity,Accuracy scores
    '''
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scorers={'sensitivity': make_scorer(recall_score) }
    supervised_result=[]
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameter_dict, 
                               n_jobs=-1, cv=cv,
                               scoring = scorers
                               ,error_score=0,refit='sensitivity')
    grid_result = grid_search.fit(X, y)

    best_model = grid_result.best_estimator_
    y_hat=best_model.predict_proba(X)
    y_pred = np.zeros(len(y_hat))
    for i in range(len(y_hat)):
      if y_hat[i, 1] > threshold:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
        y_pred = y_pred.astype(int)
    supervised_result=[grid_result.best_params_]+  performance_metrics(y,y_pred)
    return best_model,supervised_result

def test_model_basic(y,y_pred):
    '''
    Function to find the predicted labels on test set given a trivial or a baseline model
    
    Parameters:
    y- target label
    y_pred- predicted label

    Return:
    List of performance metrics Specificity,Sensitivity,Accuracy scores
    '''
    conf=confusion_matrix(y, y_pred, labels = [0, 1])
    tn, fp, fn, tp = conf[0][0],conf[0][1],conf[1][0],conf[1][1]
    specificity = (tn / (tn+fp)).round(2)
    sensitivity=(tp/(tp+fn)).round(2)
    acc=(accuracy_score(y,y_pred)).round(2)
    colormap = sns.color_palette("Greens")
    plt.figure(figsize=(6,6))  
    sns.heatmap(conf, annot=True, fmt='g', cmap=colormap)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return [specificity,sensitivity,acc]

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# SL: Trivial Model
class Trivial_Model:
    def __init__(self,X_train_final,y_train): 
        self.X_train =X_train_final
        self.y_train =y_train

    def train_trivial(self,name):# training Trivial Model
        len_train_data = len(self.y_train)
        result_train=[]
        N1=(self.y_train == 0).sum() #total number of points belonging to class S1
        N2=(self.y_train == 1).sum() #total number of points belonging to class S2
        N=N1+N2
        p1=N1/N #probability of point belonging to class S1
        p2=N2/N #probability of point belonging to class S2
        y_pred = np.random.binomial(1, p1, len_train_data) #Generating class labels with appropriate probabilities
        values=test_model_basic(self.y_train,y_pred)
        result_train=[name]+values
        return result_train,p1
        

    def test_trivial(self,name,p1,X_test_final,y_test):#testing the model
        result_test=[]
        y_pred = np.random.binomial(1, p1,  len(X_test_final)) #Generating class labels with appropriate probabilities
        values= test_model_basic(y_test,y_pred)
        
        result_test=[name]+values
        return result_test
###########################################################################################################################
# SL: Baseline Model
class Baseline_Model:
    def __init__(self,X_train_final,y_train): 
        self.X_train =X_train_final
        self.y_train =y_train

    def train_baseline(self,name):# training Baseline Model
        baseline_model = NearestCentroid() 
    
        baseline_model.fit(self.X_train,self.y_train) #fitting the Nearest means model on train data
        y_pred = baseline_model.predict(self.X_train) #predicting accuracy on train data
        values=test_model_basic(self.y_train,y_pred)
        result_train=[name]+values
        return baseline_model, result_train

    def test_baseline(self,name,best_baseline_model,X_test_final,y_test):#testing the model
        result_test=[]
        y_pred = best_baseline_model.predict(X_test_final)
        values= test_model_basic(y_test,y_pred)

        result_test=[name]+values
        return result_test
###########################################################################################################################

# SL: Logistic Regression
class Logisitic_Regression:
    
    def __init__(self,log_reg_clf,log_reg_param_grid,X_train_final,y_train): 
        self.log_reg_clf =log_reg_clf
        self.log_reg_param_grid =log_reg_param_grid
        self.X_train_final =X_train_final
        self.y_train =y_train
        # self.best_model = None
    
    def train_log_reg(self,name,threshold_train_value):# training Logistic Regression and finding the optimal hyperparamters using grid search
        result_train=[]
        best_model,values= supervised_training_model(self.log_reg_clf,
                                              self.log_reg_param_grid,
                                              self.X_train_final,
                                              self.y_train,threshold_train_value)
        result_train=[name]+values
        return best_model,result_train

    def test_log_reg(self,name,model,X_test_final,y_test,threshold_value):#testing the model
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test
    
        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test

###########################################################################################################################

# SL: Random Forest

class Random_Forest:
    
    def __init__(self,rf_clf,rf_param_grid,X_train_final,y_train): 
        self.rf_clf =rf_clf
        self.rf_param_grid =rf_param_grid
        self.X_train_final =X_train_final
        self.y_train =y_train

    
    def train_rf(self,name,threshold_train_value):# training Random Forest and finding the optimal hyperparamters using grid search
        result_train=[]
        best_model,values= supervised_training_model(self.rf_clf,
                                              self.rf_param_grid,
                                              self.X_train_final,
                                              self.y_train,threshold_train_value)
        result_train=[name]+values
        return best_model,result_train

    def test_rf(self,name,model,X_test_final,y_test,threshold_value):#testing the model
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test

###########################################################################################################################

# SL: Support Vector Classifier


class Support_Vector_Classifier:
    
    def __init__(self,svc_clf,svc_param_grid,X_train_final,y_train): 
        self.svc_clf =svc_clf
        self.svc_param_grid =svc_param_grid
        self.X_train_final =X_train_final
        self.y_train =y_train

    
    def train_svc(self,name,threshold_train_value):# training Support Vector Classifier and finding the optimal hyperparamters using grid search
        result_train=[]
        best_model,values= supervised_training_model(self.svc_clf,
                                              self.svc_param_grid,
                                              self.X_train_final,
                                              self.y_train,threshold_train_value)
        result_train=[name]+values
        return best_model,result_train

    def test_svc(self,name,model,X_test_final,y_test,threshold_value):#testing the model
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test

###########################################################################################################################

# SL: Multi Layer Perceptron


class Multi_Layer_Perceptron:
    
    def __init__(self,mlp_clf,mlp_param_grid,X_train_final,y_train): 
        self.mlp_clf =mlp_clf
        self.mlp_param_grid =mlp_param_grid
        self.X_train_final =X_train_final
        self.y_train =y_train

    
    def train_mlp(self,name,threshold_train_value):# training Multi Layer Perceptron and finding the optimal hyperparamters using grid search
        result_train=[]
        best_model,values=  supervised_training_model(self.mlp_clf,
                                              self.mlp_param_grid,
                                              self.X_train_final,
                                              self.y_train,threshold_train_value)
        result_train=[name]+values
        return best_model,result_train

    def test_mlp(self,name,model,X_test_final,y_test,threshold_value):#testing the model
        result_test=[]
        self.X_test_final =X_test_final
        self.y_test =y_test

        values= test_model(model,
                    self.X_test_final,
                    self.y_test,threshold_value)

        result_test=[name]+values
        return result_test

###########################################################################################################################

# SL: Decision Tree

class Decision_Tree:
    
    def __init__(self,dec_tree_clf,dec_tree_param_grid,X_train_final,y_train): 
        self.dec_tree_clf =dec_tree_clf
        self.dec_tree_param_grid =dec_tree_param_grid
        self.X_train_final =X_train_final
        self.y_train =y_train

    
    def train_dec_tree(self,name,threshold_train_value):# training Decision Tree and finding the optimal hyperparamters using grid search
        result_train=[]
        best_model,values=  supervised_training_model(self.dec_tree_clf,
                                              self.dec_tree_param_grid,
                                              self.X_train_final,
                                              self.y_train,threshold_train_value)

        result_train=[name]+values
        return best_model,result_train

    def test_dec_tree(self,name,model,X_test_final,y_test,threshold_value):#testing the model
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

