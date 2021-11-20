# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sklearn.metrics as skm
from sklearn.model_selection import KFold


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
        H=(M+1)*D_
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
    
def main(x_train, x_test, y_train, y_test, M, alpha, beta):
    # Parameters for prior distribution
    D = x_train.shape[1] # number of dimensions
    H = (M+1)*D # number of dimensions after transformation 
    mu_0 = np.zeros(H)
    Sigma_0 = np.identity(H)
    

    valid_scores=k_fold_cross_validation(x_train, y_train, M, mu_0, Sigma_0, alpha, beta)
    print(f'CV score: {np.mean(valid_scores)}')
    
    # Calculate posterior
    BayesLR = BayesianLinearRegression()
    BayesLR.set_prior(mu_0, Sigma_0, alpha, beta)
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
    plt.scatter(pred_mean, y_test,  color='black', marker='o')
    plt.xlabel("estimate_y")
    plt.ylabel("real_y")
    #plt.ylim(-2,2.5)
    #plt.xlim(-2,2.5)
    plt.legend()
    plt.show()
    
    #display scores for this regression model
    MAE=skm.mean_absolute_error(y_test, pred_mean)
    MSE=skm.mean_squared_error(y_test, pred_mean)
    RMSE=np.sqrt(MSE)
    r2=skm.r2_score(y_test, pred_mean)
    
    print("MSE=%f" % MSE)
    print("RMSE=%f" % RMSE)
    print("r2_score=%f" % r2)
    
    #output w_  and scores
    
    
    '''
    #近似曲線描画用の格子点生成
    xg=np.arange(-2, 2, 0.1)
    yg=np.arange(-2, 2, 0.2)
    X,Y = np.meshgrid(xg, yg)
    pred_mean=np.zeros(shape=(xg.shape[0], yg.shape[0]))
    pred_var=np.zeros(shape=(xg.shape[0], yg.shape[0]))
    # Calculate posterior
    BayesLR.learn(x_train, y_train, M)
    # Learn and predict one sample at a time
    for i in range(len(xg)):
        for j in range(len(yg)):
            

            # Calculate predictive
            X_test=np.zeros(2)
            X_test[0]=xg[i]
            X_test[1]=yg[j]
            pred_mean[i,j], pred_var[i,j] = BayesLR.predict(X_test.reshape(-1,1).T)
    upper = pred_mean + np.sqrt(pred_var)
    lower = pred_mean - np.sqrt(pred_var)
    
    
    #2D plot
    plt.plot(x_lin, pred_mean, color='orangered', linewidth = 2.5, linestyle='dashed', label='predictive mean')
    plt.fill_between(x_lin[:,:], upper, lower, color='lightgrey', label='predictive variance')
    plt.scatter(x_train[0:i+1,:], y_train[0:i+1], color='black', marker='o')
    plt.ylim(-2.5,2.5)
    plt.title('N = {}'.format(i+1))
    plt.show()
    
        

        

    fig = plt.figure() #プロット領域の作成
    ax = fig.gca(projection='3d') #プロット中の軸の取得。gca は"Get Current Axes" の略。

    #ax.plot_surface(X, Y, upper.reshape(-1,1))
    #ax.plot_surface(X, Y, lower.reshape(-1,1))
    ax.scatter(x_train[:,0], x_train[:,1], y_train, color='black', marker='o')
    ax.plot_surface(X, Y, pred_mean.T, color='orangered',alpha=0.5, label='predictive mean')
    plt.show()
    '''
def k_fold_cross_validation(X, y, M, mu_0, Sigma_0, alpha, beta):
    FOLD = 5

    valid_scores = []
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

    
    for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
        X_train, X_test = X[train_indices], X[valid_indices]
        y_train, y_test = y[train_indices], y[valid_indices]

        BayesLR = BayesianLinearRegression()
        BayesLR.set_prior(mu_0, Sigma_0, alpha, beta)
        
        BayesLR.learn(X_train, y_train, M)
        pred_mean, pred_var= BayesLR.predict(X_test)
        score=skm.r2_score(y_test, pred_mean.T)
        valid_scores.append(score)
    return valid_scores

if __name__=="__main__":
    
    # データセットの読み込み形式np.array
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=123)

    #standardize    

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
    
    #hyper_parameters
    
    M = 2 # degree of polynimial
    alpha=1
    beta=1
    
    #optimize
    for M in range(1,5):
        print("M=%d" % M)
        main(x_train, x_test, y_train, y_test, M, alpha, beta)
    

