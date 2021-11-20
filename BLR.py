# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

class BayesianLinearRegression:
    """Bayesian linear regression class"""
    def __init__(self):
        """Self variables
        
        self.mu_0 (Dx1 column vector): Mean vector
        self.Sigma_0 (DxD matrix): Covariance matrix
        """
        self.mu_0 = None
        self.Sigma_0 = None
        self.alpha=None
        self.beta=None
        self.w_=None
        
    def set_prior(self, mu_0, Sigma_0, alpha, beta):
        """Set parameters for prior distribution
        """
        self.mu_0 = mu_0 if mu_0.shape[0] == 1 else mu_0.T
        self.Sigma_0 = Sigma_0
        self.alpha=alpha
        self.beta=beta
        
    def learn(self, x_train, y_train, M):
        """Learn model and update posterior distribution
        """
        N_ = x_train.shape[0]
        D_ = x_train.shape[1]
        H=(M+1)*D
        phi_ = self.base_polynomial(x_train, M)

        Sigma_ = np.linalg.inv(self.alpha*np.identity(H)+self.beta*np.dot(phi_.T, phi_))
        mu_ = self.beta*np.dot(Sigma_, np.dot(phi_.T, y_train))
        
        self.mu_0 = mu_
        self.Sigma_0 = Sigma_
        
        return mu_, Sigma_
        
    def get_sample_from_posterior(self, x_lin):
        """Get sample from posterior distiburion
        """
        w_ = np.random.multivariate_normal(mean=self.mu_0.flatten(), cov=self.Sigma_0)

        phi_ = self.base_polynomial(x_lin, M).T
        sample_mean_ = np.einsum('h,hn->n', w_, phi_)
        
        return sample_mean_
        
    def predict(self, x_test):
        """Calculate predictive distribution
        """
        phi_ = self.base_polynomial(x_test, M)
        y_est_ = np.dot(self.mu_0.T, phi_.T)
        sigma_y_est_ = self.beta**(-1) + np.dot(phi_, np.dot(self.Sigma_0, phi_.T))
        
        return y_est_, sigma_y_est_
    
    def base_polynomial(self, x, M):
        """Calculate polynomial basis function
        """

        N_ = x.shape[0]
        D_ = x.shape[1]
        phi_ = np.zeros(shape=(N_, int((M+1)*D_)), dtype='float')
        for m in range((M+1)):
            phi_[:,m::(M+1)] = x**m
        return phi_
    
if __name__=="__main__":
    
    # ボストンデータセットの読み込み
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=123)
    
    scaler_x = StandardScaler()
    scaler_x.fit(x_train)
    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)
    
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    y_train=scaler_y.transform(y_train)
    y_test=scaler_y.transform(y_test)
    
    #
    
    D = x_train.shape[1] # number of dimensions
    M = 5 # degree of polynimial
    H = (M+1)*D # number of dimensions after transformation 
    
    # Parameters for prior distribution
    mu_0 = np.zeros(H)
    Sigma_0 = np.identity(H)
    alpha=0.1
    beta=0.1
    
    
    BayesLR = BayesianLinearRegression()
    BayesLR.set_prior(mu_0, Sigma_0, alpha, beta)
    

    # Calculate posterior
    BayesLR.learn(x_train, y_train, M)

    # Calculate predictive

    pred_mean, pred_var= BayesLR.predict(x_test)
    pred_mean=pred_mean.flatten()
    upper = pred_mean + np.sqrt(pred_var.mean(axis=1))
    lower = pred_mean - np.sqrt(pred_var.mean(axis=1))
        
    R_data=pd.DataFrame(np.stack([pred_mean, upper, lower]).T, columns=["estimate_y", "upper", "lower"])
    R_data1=R_data.sort_values("estimate_y")

    #2D plot
    plt.plot(R_data1["estimate_y"], R_data1["estimate_y"], color='orangered', linewidth = 2.5, linestyle='dashed', label='predictive mean')
    plt.fill_between(R_data1["estimate_y"], R_data1["upper"], R_data1["lower"], color='lightgrey', label='predictive variance')
    plt.scatter(R_data1["estimate_y"], y_test,  color='black', marker='o')
    plt.ylim(-2,2.5)
    plt.xlim(-2,2.5)
    plt.show()

        

