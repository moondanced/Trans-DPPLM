## Algorithm 3 for DP-FL
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# import os
from group_lasso import GroupLasso
import patsy
import pandas as pd

def noisy_ht(v, s, epsilon, delta, lambda_):
    """
    Noisy Hard Thresholding Algorithm (NoisyHT).
    
    Parameters:
    v (np.ndarray): Input vector of shape (p,).
    s (int): Sparsity parameter.
    epsilon (float): Privacy parameter.
    delta (float): Privacy parameter.
    lambda_ (float): Sensitivity.
    
    Returns:
    np.ndarray: The output of the NoisyHT algorithm.
    """
    p = len(v)
    S = set()
    scale = lambda_ * (2 * np.sqrt(3 * s * np.log(1 / delta)) / epsilon) / 20
    v_temp = 1*v
    
    for i in range(s):
        w_i = 1 * np.random.laplace(0, scale, size=p)
        j_star = np.argmax(np.abs(v_temp) + w_i)
        S.add(j_star)
        # S.add(i)
        v_temp = 1*v
        v_temp[list(S)] = 0
    
    tilde_w = 1e-0 * np.random.laplace(0, scale, size=p)
    
    v_tilde = v + tilde_w
    output = np.zeros(p)
    output[list(S)] = v_tilde[list(S)]
    # print(list(S))
    return output

def coef_gen(s, h, q=30, size_A0=0, M=20, sig_beta=0.3, sig_delta1=0.3, sig_delta2=0.5, p=500, exact=True):
    """
    Generates coefficient matrix W and initial beta vector beta0 for a given set of parameters.
    Parameters:
    s (int): Number of non-zero elements in the initial beta vector.
    h (int): Number of elements to be adjusted by sig_delta1 when k <= size_A0.
    q (int, optional): Number of elements to be adjusted by sig_delta2 when k > size_A0. Default is 30.
    size_A0 (int, optional): Threshold for determining which delta value to use. Default is 0.
    M (int, optional): Number of columns in the coefficient matrix W. Default is 20.
    sig_beta (float, optional): Standard deviation for the initial beta vector. Default is 0.3.
    sig_delta1 (float, optional): Adjustment value for elements when k <= size_A0. Default is 0.3.
    sig_delta2 (float, optional): Adjustment value for elements when k > size_A0. Default is 0.5.
    p (int, optional): Total number of elements in the beta vector and rows in the coefficient matrix W. Default is 500.
    exact (bool, optional): Flag to determine whether to use exact sampling or normal distribution for adjustments. Default is True.
    Returns:
    tuple: A tuple containing:
        - W (numpy.ndarray): Coefficient matrix of shape (p, M).
        - beta0 (numpy.ndarray): Initial beta vector of length p.
    """
    beta0 = np.concatenate((np.full(s, sig_beta), np.zeros(p - s)))
    W = np.repeat(beta0.reshape(-1, 1), M, axis=1)
    for k in range(1, M):
        if k <= size_A0:
            if exact:
                samp0 = np.random.choice(range(p), h, replace=False)
                W[samp0, k-1] += np.full(h, -sig_delta1)
            else:
                W[:100, k-1] += np.random.normal(0, h/100, size=100)
        else:
            if exact:
                samp1 = np.random.choice(range(p), q, replace=False)
                W[samp1, k-1] += np.full(q, -sig_delta2)
            else:
                W[:100, k-1] += np.random.normal(0, q/100, size=100)
    return W, beta0

def max_eigen(A, num_iter=100):
    """幂迭代估计矩阵 A 的最大特征值（返回 lambda_max）"""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = A.dot(v)
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0
        v = v / norm
    rayleigh = v.dot(A.dot(v))
    return rayleigh

def dp_estimate_w(Sig_hat, eta_0, epsilon, delta, B, T, R, C, s_star, w_0, j, n_local):
    """
    修正版 dp_estimate_w：显式传入 n_local（用于噪声标度），用传入 eta_0 作为最大允许步长候选值。
    Sig_hat: (p,p)
    n_local: 用于噪声标定的样本数（例如 n0）
    返回: w_t (p,)
    """
    p = Sig_hat.shape[0]
    w_t = w_0.copy()
    # 选择稳定步长：eta <= 1 / lambda_max
    # lambda_max = max_eigen(Sig_hat, num_iter=50)
    # if lambda_max > 0:
    #     eta = min(eta_0, 1.0 / (1.1 * lambda_max))  # 1.1 做一点保守
    # else:
    #     eta = eta_0
    eta = eta_0 / 2
    # 噪声标度函数（每轮）
    for t in range(2*T):
        w_t_half = w_t + eta * (np.eye(p)[j] - Sig_hat @ w_t)
        # 对 w_t_half 做带隐私的 hard thresholding：将 noisy_ht 中 n 改为显式 n_local
        w_t = np.clip(noisy_ht(w_t_half, s_star, epsilon / T, delta / T, eta * B / n_local), -C, C)
    return w_t

def peeling(v, s, epsilon, delta, lambda_):
    """
    Peeling Algorithm.
    
    Parameters:
    v (np.ndarray): Input vector of shape (p,).
    s (int): Sparsity parameter.
    epsilon (float): Privacy parameter.
    delta (float): Privacy parameter.
    lambda_ (float): Sensitivity.
    
    Returns:
    np.ndarray: The output of the Peeling algorithm.
    """
    p = len(v)
    S = []

    # Generate noise for the initial selection process
    for j in range(s):
        noise = np.random.laplace(0, 2 * lambda_ * np.sqrt(3 * s * np.log(1 / delta)) / epsilon, p)
        available_indices = list(set(range(p)) - set(S))
        noisy_values = np.abs(v[available_indices]) + noise[available_indices]
        j_star = available_indices[np.argmax(noisy_values)]
        S.append(j_star)
    
    # Generate noise for the output vector
    w = np.random.laplace(0, 2 * lambda_ * np.sqrt(3 * s * np.log(1 / delta)) / epsilon, p)
    # Construct the output vector
    v_tilde = np.zeros(p)
    for j in S:
        v_tilde[j] = v[j] + w[j]
    
    return v_tilde

def clipped_gradient_batch(X, Y, beta, R_t_k, R): 
    X = np.asarray(X) 
    Y = np.asarray(Y) 
    beta = np.asarray(beta)
    res = X.dot(beta) - Y 
    res_clipped = np.clip(res, -R_t_k, R_t_k) 
    X_clipped = np.clip(X, -R, R) 
    gradient_k = (res_clipped[:, None] * X_clipped).mean(axis=0) 
    
    return gradient_k 

def hard_thresholding(beta, s_prime):
    """
    Apply hard thresholding to the input array `beta`.
    Hard thresholding sets all but the largest `s_prime` absolute values in `beta` to zero.
    Parameters:
    beta (numpy.ndarray): The input array to be thresholded.
    s_prime (int): The number of largest absolute values to retain in `beta`.
    Returns:
    numpy.ndarray: The thresholded array with only the `s_prime` largest absolute values retained.
    """

    threshold_indices = np.argsort(np.abs(beta))[-s_prime:]
    beta_thresholded = np.zeros_like(beta)
    beta_thresholded[threshold_indices] = beta[threshold_indices]
    return beta_thresholded

def residualize_by_Bs(X, B_s):
    """
    批量残差化 X 对 B_s： X_tilda = X - P_{B_s} X
    输入:
      X: (n, p)
      B_s: (n, m)
    返回:
      X_tilda: (n, p)
    """
    # QR 分解 (economy)
    Q, R = np.linalg.qr(B_s, mode='reduced')   # Q:(n,m), R:(m,m)
    # 投影
    Proj = Q @ (Q.T @ X)                       # (n,p)
    X_tilda = X - Proj
    return X_tilda

# 使用示例（与原 simulate 中相同变量名）
# X_tilda = residualize_by_Bs(X, B_s)

def fit_group_lasso_and_residualize(X, B_s, Y, group_lasso_model=None):
    """
    先用 group-lasso 拟合 B_s 上的样条系数（如果需要），然后残差化 X 和 Y。
    返回 X_tilda, Y_tilda, coef_spline
    """
    if group_lasso_model is None:
        # 这里保持与原来类似的 GroupLasso API 以便替换
        from group_lasso import GroupLasso
        group_lasso_model = GroupLasso(
            groups = np.concatenate([np.arange(0), np.ones(B_s.shape[1]) + 0]),
            group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False,
            scale_reg="inverse_group_size", n_iter=1000, tol=1e-3,
            fit_intercept=False, random_state=42, warm_start=True, supress_warning=True
        )

    # 拟合样条基到 Y，得到样条系数
    group_lasso_model.fit(B_s, Y)
    coef_spline = group_lasso_model.coef_.copy()  # (m,)
    # 残差化 Y
    Y_resid = Y - (B_s @ coef_spline).flatten()

    # 批量残差化 X: X_tilda = X - P_{B_s} X
    # 用 QR
    Q, _ = np.linalg.qr(B_s, mode='reduced')    # Q:(n,m)
    Proj = Q @ (Q.T @ X)                        # (n,p)
    X_tilda = X - Proj

    return X_tilda, Y_resid, coef_spline

def federated_high_dimensional_linear_regression(datasets, T, rho, epsilon, delta, beta_init, eta, s_prime, L):
    """
    This function implements a federated learning algorithm for high-dimensional linear regression. 
    It performs distributed training across multiple clients, each with their own dataset, and 
    aggregates the results to update the model parameters.
    Parameters:
    datasets (list of tuples): A list where each element is a tuple (X_k, Y_k) representing the dataset 
                               of the k-th client. X_k is a 2D numpy array of shape (n_k, p) containing 
                               the feature vectors, and Y_k is a 1D numpy array of shape (n_k,) containing 
                               the target values.
    T (int): The number of iterations for the training process.
    rho (float): The learning rate for the gradient descent updates.
    epsilon (float): The privacy parameter for differential privacy.
    delta (float): The privacy parameter for differential privacy.
    beta_init (numpy array): The initial value of the regression coefficients.
    eta (float): A parameter used in the calculation of the clipping bounds.
    s_prime (int): The sparsity level for the hard thresholding step.
    L (float): A parameter used in the calculation of the clipping bounds.
    Returns:
    numpy array: The final value of the regression coefficients after training.
    """
    def federated_one_round_aggregate(datasets, beta, R, eta, epsilon_round, delta_round, accountant=None):
        """
        在一轮内完成所有客户端的向量化梯度计算、加噪并比例聚合（返回 Z_t）。
        datasets: list of (X_k, Y_k)
        beta: 当前模型参数
        返回: Z_t (p,)
        """
        N = sum(len(Xk) for Xk, _ in datasets)
        p = beta.shape[0]
        Z_t = np.zeros(p)
        for (X_k, Y_k) in datasets:
            n_k = len(Y_k)
            # 这里把客户端切成一个 batch（或按原来 b_k 的逻辑）
            b_k = max(1, n_k // 1)  # 若想像原代码按 T 划分，需要传入 t
            # 计算 R_t_k: 这里示例以 sqrt(log(n_k/eta)) 的形式
            R_t_k = 2.0 * np.sqrt(np.log(max(2, n_k) / (1e-6)))  # 占位

            # 向量化梯度
            gradient_k = clipped_gradient_batch(X_k[:b_k, :], Y_k[:b_k], beta, R_t_k, R)
            # 噪声（按 L2 灵敏度）
            Delta2 = (2.0 * R * R_t_k) / b_k
            sigma = Delta2 * np.sqrt(2.0 * np.log(1.25 / delta_round)) / epsilon_round
            w_t_k = np.random.normal(0.0, sigma, size=p)
            # 聚合并权重化
            Z_k_t = (n_k / N) * (gradient_k + w_t_k)
            Z_t += Z_k_t
            if accountant is not None:
                accountant.add_mechanism(epsilon_round, delta_round, name="client_grad_noisy")
        return Z_t

    # num_clients = len(datasets)
    N = sum(len(dataset[0]) for dataset in datasets)
    p = len(datasets[0][0][0])
    R = 2 * np.sqrt(L * p * np.log(N / eta))
    epsilon_prime = epsilon / 2
    delta_prime = delta / 2
    beta = beta_init
    
    for t in range(T):
        Z_t = np.zeros(p)
        
        for X_k, Y_k in datasets:
            n_k = len(Y_k)
            b_k = n_k // T
            tau_k = b_k * t
            R_t_k = 2 * np.sqrt(np.log(n_k / eta)) #* private_variance([np.dot(X_k[tau_k + i], beta) - Y_k[tau_k + i] for i in range(b_k)], epsilon_prime, delta_prime)
            
            gradient_k = np.zeros(p)
            for i in range(b_k):
                x_i = X_k[tau_k + i]
                y_i = Y_k[tau_k + i]
                # gradient_k += project_to_ball(np.dot(x_i, beta) - y_i, R_t_k) * project_to_ball(x_i, R)
                gradient_k += np.clip(np.dot(x_i, beta) - y_i, -R_t_k, R_t_k) * np.clip(x_i, -R, R)
            
            gradient_k /= b_k
            # gradient_k = (gradient_k-np.mean(gradient_k))/np.sqrt(np.var(gradient_k))
            w_t_k = np.random.normal(0, np.sqrt(8 * np.log(2.5 / delta_prime) / (b_k**2 * epsilon_prime**2)), p)
            # print(np.max(w_t_k))
            Z_k_t = n_k / N * (gradient_k + w_t_k)
            Z_t += Z_k_t
        # Z_t = (Z_t-np.mean(Z_t))/np.sqrt(np.var(Z_t))
        # Z_t = federated_one_round_aggregate(datasets, beta, R, eta, epsilon / T, delta / T, None)
        beta_half = beta - rho * Z_t
        beta = hard_thresholding(beta_half, s_prime)

    return beta

import time

start = time.perf_counter()
# Parameters
np.random.seed(42)
n = 1000
n0 = 1000
p = 1000
d = 2
h0 = 4
rho = 5e-1
# epsilon = 1e-5
epsilon = 0.6
# delta = 1e-5
delta = 1/(n**1.1)
eta = 0.1 
s = 10
s_prime = 15
L = 1.0 
Nsim = 500

corr = 0.6
A_sizes = [3, 5, 8, 10, 12, 15]
# A_sizes = [12, 15, 20]
parameter_list = {
    'n': n,
    'n0': n0,
    'p': p,
    'd': d,
    'h': h0,
    'rho': rho,
    'epsilon': epsilon, 
    'delta': delta,
    'eta': eta,
    's': s,
    's_prime': s_prime,
    'L': L,
    'Nsim': Nsim,
    'corr': corr,
    'A_sizes': A_sizes
}

idx = np.linspace(1,p,p)
idx1, idx2 = np.meshgrid(idx, idx)
Sig_0 = corr**(np.abs(idx1-idx2))
Sig_k = (corr + 0)**(np.abs(idx1-idx2))
# Sig = corr*np.ones((p,p))
# Sig += (1-corr)*np.eye(p)
# print(np.linalg.inv(Sig_0))

# Range of sample sizes
MSE_init_all = []
MSE_de_all = []
MSE_de_dp_all = []
Bias_de_all = []
Bias_de_dp_all = []
inrate_all = []
cilen_all = []
e_value_all = []
p_value_all = []
beta_true_all = []
parameter_all = []

def simulate(num_clients):
    w_0 = np.zeros(p)
    B = 1.0
    R = 2 * np.log(n)
    C = 1.0
    T = 1 * int(np.ceil(np.log(n)))
    R = 2 * np.log(n)


    # Simulation
    MSE_init = []
    MSE_de = []
    MSE_de_dp = []
    Bias_de = np.zeros([p, Nsim])
    Bias_de_dp = np.zeros([p, Nsim])
    inrate = np.zeros([p,Nsim])
    cilen = np.zeros([p,Nsim])
    e_value = np.zeros([p,Nsim])
    # e_value_noisy = np.zeros([p,Nsim])
    p_value = np.zeros([p,Nsim])
    IR_S = np.zeros(Nsim)
    IR_Sc = np.zeros(Nsim)

    for l in range(Nsim):
        datasets = []
        coef_all, beta_true = coef_gen(s, h=h0, q=2*s, size_A0=num_clients, M=num_clients, sig_beta=0.5,
                            sig_delta1=0.5, sig_delta2=(0.7), p=p, exact=True)
        coef_all = coef_all / 0.5 
        for m in range(num_clients-1):
            X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sig_k, size=n)
            Y = X @ coef_all[:,m] + np.random.randn(n)
            datasets.append((X, Y))
        beta_true = coef_all[:, num_clients-1]
        # beta_true[beta_true!=0] += np.random.randn(np.sum(beta_true!=0))/50
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sig_0, size=n0)
        # X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n0)
        # X = np.random.uniform(0,1,(n0, p))
        Y = X @ beta_true + np.random.randn(n0)
        # print(np.mean((Y - np.dot(X, beta_true)) ** 2))
        z = np.random.uniform(0,1,(n0, d))
        # z = X[:,0:d]
        # z[:,0] = np.abs(X[:,0]+X[:,1])
        # z[:,1] = np.abs(X[:,0]-X[:,1])

        design_matrices = []
        z_df = pd.DataFrame(z, columns=[f'z_{i}' for i in range(d)])
        for i in range(d):
            spline_formula = f"0 + bs(z_{i}, df=5)"
            design_matrix = patsy.dmatrix(spline_formula, z_df, return_type='dataframe').values
            design_matrices.append(design_matrix)
        B_s = design_matrices[0]
        for i in range(len(design_matrices)-1):
            B_s = np.hstack((B_s, design_matrices[i+1]))
        n_spline_features = B_s.shape[1]
        groups = (np.concatenate([np.arange(p), np.ones(n_spline_features) + p]))

        for i in range(d):
            Y = Y + 4 * np.sin(2 * np.pi * z_df[f'z_{i}']+ i * np.pi / 2)
            # Y = Y + 1/(1+np.exp(-z_df[f'z_{i}'])/d)
        group_lasso = GroupLasso(groups=groups, group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False, scale_reg="inverse_group_size", n_iter=2000, tol=1e-3, fit_intercept=False, random_state=42, warm_start=True, supress_warning=True)
        
        X_combined = np.hstack((X, B_s))
        group_lasso.fit(X_combined, Y)
        coef_spline = group_lasso.coef_[p:]
        Y_raw = Y.copy()
        Y = Y - (B_s @ coef_spline).flatten()
        datasets.append((X, Y))
        beta_init = np.zeros(p)
        beta_final = federated_high_dimensional_linear_regression(datasets, T, rho, epsilon, delta, beta_init, eta, s_prime, L)
        # beta_final = beta_true

        beta_de = np.zeros(p)
        res_std = np.zeros(p)
        beta_de_dp = np.zeros(p)
        res_std_dp = np.zeros(p)
        groups = (np.concatenate([np.arange(0), np.ones(n_spline_features) + 0]))
        group_lasso = GroupLasso(groups=groups, group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False, scale_reg="inverse_group_size", n_iter=1000, tol=1e-3, fit_intercept=False, random_state=42, warm_start=True, supress_warning=True)
        # X_combined = np.hstack((X, B_s))
        # print(Y.copy() )
        group_lasso.fit(B_s, Y_raw - np.dot(X, beta_final))
        coef_spline = group_lasso.coef_
        Y = Y_raw - (B_s @ coef_spline).flatten()
        clipped_Y = np.clip(Y, -R, R)
        clipped_X_beta_final = np.clip(np.dot(X, beta_final), -R, R)
        residuals = clipped_Y - clipped_X_beta_final
        
        groups = (np.concatenate([np.arange(0), np.ones(n_spline_features) + 0]))
        group_lasso = GroupLasso(groups=groups, group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False, scale_reg="inverse_group_size", n_iter=1000, tol=1e-3, fit_intercept=False, random_state=42, warm_start=True, supress_warning=True)
        X_tilda = X.copy()
        for i in range(p):
            # X_j = X[:,i].reshape(-1,1)
            group_lasso.fit(B_s, X[:,i])
            X_tilda[:,i] = X[:,i] - (B_s @ group_lasso.coef_).flatten()

        sigma_hat = np.sqrt(np.sum(residuals ** 2) / n0 + 5e-2 * np.random.randn(1) * np.sqrt((4*R**2/n0) * (2*np.log(1.25/delta)/epsilon))) 
        # sigma_hat = [sigma_hat]
        Sig_hat = X_tilda.T @ X_tilda / n0
        estimated_w_mat = np.zeros((p, p))
        
        
        for j in range(p):
            # w_0 = np.linalg.inv(Sig)[j,:]
            w_0 = np.zeros(p)
            w_j_estimated = dp_estimate_w(Sig_hat, 0.2*rho, epsilon, delta, B, int(np.ceil(T)*8), 1*R, 5*C, 5, w_0, j, n0)
            # w_j_estimated = np.linalg.inv(Sig_0)[j,:]
            # print(w_j_estimated)
            # print(np.linalg.inv(Sig_0)[j,:])
            # print(w_j_estimated/np.linalg.inv(Sig_0)[j,:])
            # w_j_estimated = 1
            estimated_w_mat[j, :] = w_j_estimated
            z_j = np.random.randn(1) * np.sqrt((4/n0)**2 * (2*np.log(1.25/delta)/epsilon**2)) * 1e-0
            beta_de[j] = beta_final[j] + np.sum(np.clip(np.dot(X_tilda, w_j_estimated), -R, R) * (np.clip(Y, -R, R) - np.clip(np.dot(X, beta_final), -R, R))) / n0 
            beta_de_dp[j] = beta_de[j] + z_j[0] 
            z_critical = stats.norm.ppf(1 - (1 - 0.95) / 2)
            
            
            cilen[j,l] = 2 * z_critical * (sigma_hat[0] * np.sqrt(w_j_estimated[j])) / np.sqrt(n0)
            beta_lo = beta_de[j] - cilen[j,l]/2
            beta_up = beta_de[j] + cilen[j,l]/2

            inrate[j,l] += (beta_true[j]>beta_lo)*(beta_true[j]<beta_up)
            if beta_true[j] == 0:
                IR_Sc[l] += (beta_true[j]>beta_lo)*(beta_true[j]<beta_up)
            else:
                IR_S[l] += (beta_true[j]>beta_lo)*(beta_true[j]<beta_up)

            if w_j_estimated[j] > 0:
                res_std[j] = np.sqrt(n0) * (beta_de_dp[j] - beta_true[j]) / (sigma_hat[0] * np.sqrt(w_j_estimated[j]))
                res_std_dp[j] = np.sqrt(n0) * (beta_de_dp[j] - beta_true[j]) / (sigma_hat[0] * np.sqrt(w_j_estimated[j]))
                t_stat = np.sqrt(n0) * (beta_de_dp[j]) / (sigma_hat[0] * np.sqrt(w_j_estimated[j]))
                p_value[j,l] = (1 - stats.norm.cdf(np.abs(t_stat)))*2
                # print((sigma_hat[0] * np.sqrt(w_j_estimated[j])))
            else:
                res_std[j] = np.nan  
                res_std_dp[j] = np.nan  

        MSE_init.append(np.mean((beta_final - beta_true) ** 2))
        MSE_de.append(np.mean((beta_de - beta_true) ** 2))
        MSE_de_dp.append(np.mean((beta_de_dp - beta_true) ** 2))

        e_value[:,l] = 0.5*(np.exp(np.sqrt(n0)*(beta_de_dp)-((sigma_hat[0] * np.sqrt(w_j_estimated[j])))**2/2)+np.exp(np.sqrt(n0)*(-beta_de_dp)-((sigma_hat[0] * np.sqrt(w_j_estimated[j])))**2/2))
        # p_value[:,l] = peeling(p_value[:,l], p, epsilon / 2, delta / 2, noise_scale)
        # e_value[:,l] = 0.5*(np.exp(res_std-(1)**2/2)+np.exp((-res_std)-(1)**2/2))
        Bias_de[:, l] = res_std
        Bias_de_dp[:, l] = res_std_dp
   

    return MSE_init, MSE_de, MSE_de_dp, Bias_de, Bias_de_dp, inrate, cilen, beta_true, e_value, p_value

results = Parallel(n_jobs=-1)(delayed(simulate)(num_clients) for num_clients in A_sizes)

elapsed = time.perf_counter() - start
print(f"耗时：{elapsed:.4f} 秒")
for result in results:
    print('abaabc')
    # print(result[3])
    MSE_init, MSE_de, MSE_de_dp, Bias_de, Bias_de_dp, inrate, cilen, beta_true, e_value, p_value = result
    # print(MSE_init)
    # print(Bias_de)
    beta_true_all.append(beta_true)
    inrate_all.append(inrate)
    cilen_all.append(cilen)
    e_value_all.append(e_value)
    p_value_all.append(p_value)
    MSE_init_all.append(MSE_init)
    print(np.mean(MSE_init_all,axis=1))
    MSE_de_all.append(MSE_de)
    MSE_de_dp_all.append(MSE_de_dp)
    print(np.mean(MSE_de_dp_all,axis=1))
    Bias_de_all.append(Bias_de)
    Bias_de_dp_all.append(Bias_de_dp)
    # print(np.mean(e_value_all,axis=0))
    # print(np.mean(p_value_all,axis=0))
    p = len(e_value)
    
    e_star = -(np.linspace(1,p,p)*np.sort(-np.mean(np.array(e_value), axis = 1)))/p
    p_star = (np.linspace(1,p,p)/np.sort(np.mean(np.array(p_value+1e-10), axis = 1)))/p
    FDP_e = np.sum((e_star>20)*(beta_true==0))/max(np.sum(e_star>20),1)
    FDP_p= np.sum((p_star>20)*(beta_true==0))/max(np.sum(p_star>20),1)
    print(FDP_e)
    print(FDP_p)
    Power_e = np.sum((e_star>20)*(beta_true!=0))/np.sum(beta_true!=0)
    Power_p = np.sum((p_star>20)*(beta_true!=0))/np.sum(beta_true!=0)
    print(Power_e)
    print(Power_p)

# Save the results as .npy files
np.save('Bias_de_all_trans.npy', Bias_de_all)
np.save('Bias_de_dp_all_trans.npy', Bias_de_dp_all)
MSE_all = []
MSE_all.append(MSE_init_all)
MSE_all.append(MSE_de_all)
MSE_all.append(MSE_de_dp_all)
# MSE = [MSE_init_all, MSE_de_all, MSE_de_dp_all]
np.save('MSE_all_trans.npy', MSE_all)
np.save('IR_all_trans.npy', inrate_all)
np.save('cilen_all_trans.npy', cilen_all)
np.save('beta_true_all_trans.npy', beta_true_all)
np.save('parameter_list.npy', parameter_list)
np.save('e_values.npy', e_value_all)
np.save('p_values.npy', p_value_all)