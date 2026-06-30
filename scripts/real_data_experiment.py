# %%
import numpy as np
from scipy import stats
from group_lasso import GroupLasso
import patsy
import matplotlib.pyplot as plt
import seaborn as sns

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
        # print(v_temp)
        w_i = 1 * np.random.laplace(0, scale, size=p)
        j_star = np.argmax(np.abs(v_temp) + w_i)
        S.add(j_star)
        # S.add(i)
        v_temp = 1*v
        v_temp[list(S)] = 0
    
    tilde_w = 1 * np.random.laplace(0, scale, size=p)
    
    v_tilde = v + tilde_w
    output = np.zeros(p)
    output[list(S)] = v_tilde[list(S)]
    # print(list(S))
    return output


def dp_estimate_w(Sig_hat, eta_0, epsilon, delta, B, T, R, C, s_star, w_0, j):
    """
    Differentially Private Estimation of w_j.
    
    Parameters:
    Sig_hat (np.ndarray): Estimated covariance matrix of shape (p, p).
    eta_0 (float): Learning rate.
    epsilon (float): Privacy parameter.
    delta (float): Privacy parameter.
    B (float): Sensitivity parameter.
    T (int): Number of iterations.
    R (float): Clipping bound.
    C (float): Clipping bound.
    s_star (int): Sparsity parameter.
    w_0 (np.ndarray): Initial vector of shape (p,).
    j (int): Index of the coordinate to estimate.
    
    Returns:
    np.ndarray: The estimated vector w_j.
    """
    w_t = w_0
    # T = 500
    # eta_0 = 1e-2
    for t in range(T):
        # w_t_half = w_t + eta_0 * (np.eye(len(w_0))[j] - Sig_hat @ w_t)
        w_t_half = np.linalg.inv(Sig_hat)[:,j]
        # print(w_t_half[j])
        w_t = np.clip(noisy_ht(w_t_half, s_star, epsilon / T, delta / T, eta_0 * B / len(w_0)), -C, C)
        # print(w_t[j])
    return w_t


def dp_fl_real_data(source_datasets,
                    target_dataset,
                    T,
                    rho,
                    epsilon,
                    delta,
                    eta,
                    s_prime,
                    L):
    """
    Differentially‐Private Federated Linear Regression + De‐biasing on real data.
    
    Parameters
    ----------
    source_datasets : list of (X_k, Y_k)
        List of client‐level training data. Each X_k is (n_k × p), Y_k is (n_k,).
    target_dataset : (X, Y)
        Target sample for de‐biasing: X is (n0 × p), Y is (n0,).
    T : int
        Number of FL iterations.
    rho : float
        Learning rate for FL.
    epsilon : float
        Total privacy budget ε.
    delta : float
        Total privacy parameter δ.
    eta : float
        Small probability parameter used in clipping bounds.
    s_prime : int
        Sparsity level for hard‐thresholding in FL.
    L : float
        Lipschitz/smoothness constant for clipping radius.
    
    Returns
    -------
    beta_init   : np.ndarray, shape (p,)
        Initial coefficients (zeros).
    beta_de     : np.ndarray, shape (p,)
        De‐biased (non‐private) point estimate.
    beta_de_dp  : np.ndarray, shape (p,)
        De‐biased, private estimate (with Gaussian noise).
    cilen       : np.ndarray, shape (p,)
        95% CI length for each coefficient.
    e_value     : np.ndarray, shape (p,)
        E‐values for each coordinate test.
    """
    # unpack target data
    X_target, Y_target, Z_target = target_dataset
    n0, p = X_target.shape
    n0, d = Z_target.shape

    beta = np.zeros(p)   # beta_init
    N = sum(X.shape[0] for X, _ in source_datasets) + n0
    R = 2 * np.sqrt(L * p * np.log((N) / eta))
    epsilon_f, delta_f = epsilon/2, delta/2

    datasets = []
    for X, Y in source_datasets:
        datasets.append((np.array(X), np.array(Y)))
    design_matrices = []
    # z_df = pd.DataFrame(Z_target, columns=[f'z_{i}' for i in range(d)])
    Z_target.columns = [f'z_{i}' for i in range(d)]
    for i in range(d):
        spline_formula = f"0 + bs(z_{i}, df=5)"
        design_matrix = patsy.dmatrix(spline_formula, Z_target, return_type='dataframe').values
        design_matrices.append(design_matrix)
    B_s = design_matrices[0]
    for i in range(len(design_matrices)-1):
        B_s = np.hstack((B_s, design_matrices[i+1]))
    n_spline_features = B_s.shape[1]
    groups = (np.concatenate([np.arange(p), np.ones(n_spline_features) + p]))
    group_lasso = GroupLasso(groups=groups, group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False, scale_reg="inverse_group_size", n_iter=2000, tol=1e-3, fit_intercept=False, random_state=42, warm_start=True, supress_warning=True)
    
    X_combined = np.hstack((X_target, B_s))
    group_lasso.fit(X_combined, Y_target)
    coef_spline = group_lasso.coef_[p:]
    Y_target = np.array(Y_target - (B_s @ coef_spline)).flatten()
    # Y_target = np.array(Y_target).flatten()
    datasets.append((np.array(X_target), np.array(Y_target)))

    
    groups = (np.concatenate([np.arange(0), np.ones(n_spline_features) + 0]))
    group_lasso = GroupLasso(groups=groups, group_reg=0.2, l1_reg=0.0, frobenius_lipschitz=False, scale_reg="inverse_group_size", n_iter=1000, tol=1e-3, fit_intercept=False, random_state=42, warm_start=True, supress_warning=True)
    X_tilda = np.array(X_target).copy()
    for i in range(p):
        # X_j = X[:,i].reshape(-1,1)
        group_lasso.fit(B_s, np.array(X_target)[:,i])
        X_tilda[:,i] = np.array(X_target)[:,i] - (B_s @ group_lasso.coef_).flatten()

    # X_tilda = np.array(X_target).copy()

    for t in range(T):
        Z_t = np.zeros(p)
        for X_k, Y_k in datasets:
            n_k = X_k.shape[0]
            b_k = n_k #max(1, n_k // T)
            tau = 0 #b_k * t
            # dynamic clipping bound
            R_tk = 2 #2 * np.sqrt(np.log(n_k / eta)) * 1e4
            grad = np.zeros(p)
            plt.plot(X_k, Y_k, 'o')
            for i in range(b_k):
                xi = X_k[tau + i]
                yi = Y_k[tau + i]
                grad += np.clip(xi.dot(beta) - yi, -R_tk, R_tk) * np.clip(xi, -R, R)
                # print(xi.dot(beta), yi)
                # print(np.mean(Y_k))
            grad /= b_k
            # print(R_tk)
            # print(grad)
            # add Gaussian noise for DP
            sigma_k = np.sqrt(8 * np.log(2.5/delta_f) / (b_k**2 * epsilon_f**2))
            w_k = 1e-1 * np.random.normal(0, sigma_k, size=p)
            Z_t += (n_k / N) * (grad + w_k)
        # gradient step + hard threshold
        print(Z_t)
        beta = beta - rho * Z_t
        thresh_idx = np.argsort(np.abs(beta))[-s_prime:]
        mask = np.zeros(p, bool); mask[thresh_idx] = True
        beta = beta * mask


    beta_init = beta.copy()
    beta_final = beta.copy()

    residuals = np.clip(Y_target, -R, R) - np.clip(X_target.dot(beta_final), -R, R)
    sigma_hat = np.sqrt(np.sum(residuals ** 2) / n0 + 5e-2 * np.random.randn(1) * np.sqrt((4*R**2/n0) * (2*np.log(1.25/delta)/epsilon))) 
    Sig_hat = X_tilda.T @ X_tilda / n0

    beta_de     = np.zeros(p)
    beta_de_dp  = np.zeros(p)
    cilen       = np.zeros(p)
    e_value     = np.zeros(p)

    z_crit = stats.norm.ppf(0.975)  # 95% two‐sided

    
        
    for j in range(p):
        # w_0 = np.linalg.inv(Sig)[j,:]

        w_0 = np.zeros(p)
        # w_j_estimated = dp_estimate_w(Sig_hat, 1*rho, epsilon, delta, 1.0, int(np.ceil(T)+0), 1*R, 1e12, 50, w_0, j)
        w_j_estimated = np.linalg.inv(Sig_hat)[j,:]
        # print(w_j_estimated[j])
        # w_j_estimated = 1
        z_j = np.random.randn(1) * np.sqrt((4/n0)**2 * (2*np.log(1.25/delta)/epsilon**2)) * 1e-0
        beta_de[j] = beta_final[j] + np.sum(np.clip(np.dot(X_tilda, w_j_estimated), -R, R) * (np.clip(Y_target, -R, R) - np.clip(np.dot(X_target, beta_final), -R, R))) / n0 
        beta_de_dp[j] = beta_de[j] + z_j[0] 
        z_crit = stats.norm.ppf(1 - (1 - 0.95) / 2)
        # 3.3) CI length & E‐value
        # print(w_j_estimated[j])
        stderr_j = sigma_hat * np.sqrt(w_j_estimated[j] / n0)
        cilen[j]  = 2 * z_crit * stderr_j
        # two‐sided e‐value
        z_stat    = beta_de_dp[j] / stderr_j
        e_value[j]= 2 * np.exp(-z_stat**2 / 2)

    return beta_init, beta_de, beta_de_dp, cilen, e_value


# %%
import pandas as pd

# 1) Core processing, no thresholding or cross‑station alignment
def process_station_data(station_code,
                         aqi_file='./data/airdata.csv',
                         wea_dir='./data/dataset/wea/'):
    # — Load and parse AQI data
    data = pd.read_csv(aqi_file)
    data['datetime'] = pd.to_datetime(
        data['datetime'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )
    data = data.dropna(subset=['datetime'])
    
    AQIS = ['PM2.5','PM10','SO2','NO2','O3']
    aqi_df = (
        data
        .pivot_table(
            index='datetime',
            columns='type',
            values=station_code,
            aggfunc='mean'
        )
        .reset_index()
    )
    aqi_df['date'] = aqi_df['datetime'].dt.date
    aqi_df = aqi_df.groupby('date')[AQIS].mean().reset_index()
    
    # — Load and parse Weather data
    wea_df = pd.read_csv(f"{wea_dir}/{station_code}.csv")
    wea_df['datetime'] = pd.to_datetime(wea_df['datetime'], errors='coerce')
    wea_df = wea_df.dropna(subset=['datetime'])
    wea_df['date'] = wea_df['datetime'].dt.date
    weather_vars = ['temperature','humidity','windSpeed','windBearing']
    wea_daily = wea_df.groupby('date')[weather_vars].mean().reset_index()
    
    return aqi_df, wea_daily


# 2) Per‑station thresholding & local alignment + tuple‐building
AQI_X = ['PM10', 'SO2', 'NO2', 'O3']
AQI_Y = ['PM2.5']
# AQI_Z = ['SO2','NO2','O3']
weather_vars = ['temperature','humidity','windSpeed','windBearing']

# define per‑pollutant cutoffs
thresholds = {'PM2.5':35, 'PM10':50, 'SO2':0, 'NO2':0, 'O3':0}

# Process **source** stations
source_stations = ['1356A', '1357A', '1358A', '1361A', '1363A', '1364A', '1365A', '1390A', '1391A', '1392A', '1393A', '1394A', '1395A', '1396A']
source_station = source_stations[:15]  
source_data = []

for st in source_station:
    aqi_df, wea_df = process_station_data(st)
    
    # apply threshold filter
    mask = pd.DataFrame({
        pol: aqi_df[pol] > thr
        for pol, thr in thresholds.items()
    }).all(axis=1)
    aqi_filt = aqi_df[mask].reset_index(drop=True)
    wea_filt = wea_df[wea_df['date'].isin(aqi_filt['date'])].reset_index(drop=True)
    
    # extract X, Y
    X = aqi_filt[AQI_X]
    Y = aqi_filt[AQI_Y]
    source_data.append((X, Y))


# Process **target** station
target_station = ['1360A']
target_data = []

for st in target_station:
    aqi_df, wea_df = process_station_data(st)
    
    mask = pd.DataFrame({
        pol: aqi_df[pol] > thr
        for pol, thr in thresholds.items()
    }).all(axis=1)
    aqi_filt = aqi_df[mask].reset_index(drop=True)
    wea_filt = wea_df[wea_df['date'].isin(aqi_filt['date'])].reset_index(drop=True)
    
    # extract X, Y, Z
    X = aqi_filt[AQI_X]
    Y = aqi_filt[AQI_Y]
    Z = wea_filt[weather_vars]
    # Z = aqi_filt[AQI_Z]
    target_data.append((X, Y, Z))


# Optional sanity check
for i, (X, Y) in enumerate(source_data):
    print(f"Source {source_station[i]}: {X.shape[0]} days retained")
X0, Y0, Z0 = target_data[0]
print(f"Target {target_station[0]}: {X0.shape[0]} days retained")


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── Extract the raw target arrays ─────────────────────────────────────────
# (after your alignment and X/Y/Z unpacking)
X_target, Y_target, Z_target = target_data[0]   # shapes: (n0×p), (n0,), (n0×d)

# ─── 1. SPLIT into train / test ───────────────────────────────────────────
# Here we stratify by Y if you need balanced splits, otherwise simple random
X_tr, X_te, Y_tr, Y_te, Z_tr, Z_te = train_test_split(
    X_target, Y_target, Z_target,
    test_size=0.2,      # 20% test
    random_state=42
)

# ─── 2. RUN your DP‑FL on training only ───────────────────────────────────
beta_init, beta_de, beta_de_dp, cilen, e_value = dp_fl_real_data(
    source_datasets = source_data,
    target_dataset  = (X_tr, Y_tr, Z_tr),
    T               = 20,
    rho             = 0.001,
    epsilon         = 0.5,
    delta           = 1e-5,
    eta             = 0.1,
    s_prime         = 4,
    L               = 1.0,
)

# ─── 3. TEST PROCEDURE ─────────────────────────────────────────────────────
# Use your private de‑biased estimate `beta_de_dp` for prediction:
Y_pred = X_te.dot(beta_init)

# Compute standard regression metrics on the held‑out set:
mse = mean_squared_error(Y_te, Y_pred)
r2  = r2_score(Y_te, Y_pred)

print("=== Test Set Evaluation ===")
print(f"MSE on test set: {mse:.4f}")
print(f"R² on test set:  {r2:.4f}")

# (Optionally) you can also compute coverage of your CIs if you want:
# lower = Y_te - 0.5 * cilen  # naive symmetric CI
# upper = Y_te + 0.5 * cilen
# coverage = np.mean((Y_pred >= lower) & (Y_pred <= upper))
# print(f"Approx. 95% CI coverage: {coverage:.2%}")

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare your full list of candidate source stations
all_source_stations = ['1356A', '1363A', '1393A', '1364A', 
                       '1358A', '1357A', '1394A', '1396A', '1361A']

# This will store test MSE for each k = 1 ... len(all_source_stations)
mse_results = []

# Loop over using the first k source stations
for k in range(1, len(all_source_stations) + 1):
    # 1. BUILD source_data for first k stations
    src_data = []
    for st in all_source_stations[:k]:
        aqi_df, wea_df = process_station_data(st)
        mask = pd.DataFrame({
            pol: aqi_df[pol] > thr
            for pol, thr in thresholds.items()
        }).all(axis=1)
        aqi_filt = aqi_df[mask].reset_index(drop=True)
        wea_filt = wea_df[wea_df['date'].isin(aqi_filt['date'])].reset_index(drop=True)
        Xs = aqi_filt[AQI_X]
        Ys = aqi_filt[AQI_Y]
        src_data.append((Xs, Ys))

    # 2. BUILD target train/test
    X_t, Y_t, Z_t = target_data[0]   # assume target_data already populated
    X_tr, X_te, Y_tr, Y_te, Z_tr, Z_te = train_test_split(
        X_t, Y_t, Z_t, test_size=0.2, random_state=42
    )

    # 3. RUN DP‑FL on the training split
    beta_init, beta_de, beta_de_dp, cilen, e_value = dp_fl_real_data(
        source_datasets = src_data,
        target_dataset  = (X_tr, Y_tr, Z_tr),
        T               = 20,
        rho             = 0.001,
        epsilon         = 0.5,
        delta           = 1e-5,
        eta             = 0.1,
        s_prime         = 4,
        L               = 1.0,
    )

    # 4. PREDICT and EVALUATE on the held‑out test split
    Y_pred = X_te.dot(beta_init)   # or .dot(beta_de_dp) if you prefer the DP version
    mse_k  = mean_squared_error(Y_te, Y_pred)
    mse_results.append(mse_k)
    print(f"k = {k:<2d} source stations → MSE = {mse_k:.4f}")

# 5. PLOT MSE vs # of sources
plt.figure(figsize=(8,5))
plt.plot(range(1, len(all_source_stations)+1), mse_results, marker='o')
plt.xticks(range(1, len(all_source_stations)+1))
plt.xlabel('Number of Source Stations')
plt.ylabel('Test MSE')
plt.title('MSE with different Number of Source Stations')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# prepare your real data:
# source = [(X1, Y1), (X2, Y2), ...]
# target = (X_target, Y_target, Z_target)

beta_init, beta_de, beta_de_dp, cilen, e_value = dp_fl_real_data(
    source_datasets = source_data,
    target_dataset  = target_data[0],
    T               = 20,
    rho             = 0.01,
    epsilon         = 0.5,
    delta           = 1e-5,
    eta             = 0.1,
    s_prime         = 4,
    L               = 1.0,
)

