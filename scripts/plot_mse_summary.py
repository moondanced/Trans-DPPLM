#%%
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

base_dir = "./results"

# Fixed parameters
n = 500
n0 = 1000
p = 500

# Example: different correlation settings
# corr_list = [0.2, 0.4, 0.6, 0.8]

# If you also vary other parameters, define lists here
n_list = [500, 1000]
# p_list = [...]

# =========================
# Loop over settings
# =========================

for n in n_list:

    # Folder naming convention
    folder_name = f"n={n}, n0={n0}, p={p}"
    folder_path = os.path.join(base_dir, folder_name)

    file_path = os.path.join(folder_path, "MSE_all_trans.npy")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load data
    mse_values = np.load(file_path)

    # If mse_values is 2D (e.g., repetitions × methods), average over repetitions
    if mse_values.ndim == 2:
        mse_mean = mse_values.mean(axis=0)
    else:
        mse_mean = mse_values

    # =========================
    # Plot
    # =========================
    plt.figure()
    plt.plot(mse_mean)
    plt.xlabel("Index")
    plt.ylabel("MSE")
    plt.title(f"MSE (n={n}, n0={n0}, p={p}")

    # Format corr for filename (e.g., 0.6 -> 06)
    # corr_str = str(corr).replace(".", "")
    save_name = f"MSE_n{n}n0{n0}p{p}.png"
    save_path = os.path.join(folder_path, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

print("Done.")
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

base_dir = "./results"
n = 1000
n0 = 500
p = 1000
corr = 0.6

A_values = np.array([3, 5, 8, 10, 12, 15])

folder_name = f"n={n},n0={n0},p={p}"
folder_path = os.path.join(base_dir, folder_name)
file_path = os.path.join(folder_path, "MSE_all_trans.npy")

# =========================
# Load data
# =========================

mse_values = np.load(file_path)

print("Loaded shape:", mse_values.shape)
# Expected: (R, 3, 6)

# =========================
# Compute mean and std over repetitions
# =========================

mse_mean = mse_values.mean(axis=2)   # → shape (3, 6)
mse_std  = mse_values.std(axis=2)    # → shape (3, 6)

# =========================
# Plot
# =========================

plt.figure()

for method_idx in range(mse_mean.shape[0]):
    plt.errorbar(
        A_values,
        mse_mean[method_idx],
        yerr=mse_std[method_idx],
        marker='o',
        capsize=4,
        label=f"Method {method_idx+1}"
    )

plt.xlabel(r"$|\mathcal{A}|$")
plt.ylabel("MSE")
# plt.title(f"MSE Comparison (n={n}, n0={n0}, p={p}")
# plt.legend()

save_name = f"MSE_n{n}n0{n0}p{p}.png"
save_path = os.path.join(folder_path, save_name)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Configuration
# =========================

base_dir = "./results"
n = 1000
n0 = 1000
p = 1000
corr = 0.6
eps = '06'

# A_values = np.array([3, 5, 8, 10, 12, 15])

folder_name = f"n={n},n0={n0},eps={eps}"
folder_path = os.path.join(base_dir, folder_name)
file_path = os.path.join(folder_path, "MSE_all_trans.npy")
# para = np.load( os.path.join(folder_path, "parameter_list.npy"), allow_pickle=True )
# =========================
# Load data
# =========================
A_sizes = [3, 5, 8, 10, 12, 15]
MSE_all = np.load(file_path)


titles = [
    'MSE for initial estimation',
    'MSE for debiased estimation',
    'MSE for dp-debiased estimation'
]

methods = ['Initial', 'Debiased', 'DP-Debiased']

# Create results directory if it doesn't exist
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Prepare data for plotting
data = []
for i, method in enumerate(methods):
    for j, a_size in enumerate(A_sizes):
        for k in range(MSE_all.shape[2]):
            data.append([MSE_all[i, j, k], method, a_size])

df = pd.DataFrame(data, columns=['MSE', 'Method', 'A_size'])

# Calculate mean and std deviation
summary = df.groupby(['Method', 'A_size'])['MSE'].agg(['mean', 'std']).reset_index()

# Map A_size to evenly spaced indices
a_size_indices = list(range(len(A_sizes)))
a_size_map = dict(zip(A_sizes, a_size_indices))
summary['A_index'] = summary['A_size'].map(a_size_map)

# Plot with evenly spaced x-axis
plt.figure(figsize=(12, 8))
for method in methods:
    sub_df = summary[summary['Method'] == method]
    plt.errorbar(
        x=sub_df['A_index'],
        y=sub_df['mean'],
        yerr=sub_df['std'],
        label=method,
        marker='o',
        capsize=5,
        linestyle='-',
    )


plt.xlabel(r'$|\mathcal{A}|$', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(title='Method', loc='upper right', fontsize=20)
plt.tight_layout()

save_name = f"MSE_n{n}n0{n0}eps{eps}.png"
save_path = os.path.join(folder_path, save_name)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Configuration
# =========================

base_dir = "./results"
n = 1000
n0 = 1000
p = 1000
corr = 0.6
eps = '04'

# A_values = np.array([3, 5, 8, 10, 12, 15])

folder_name = f"n={n},n0={n0},p={p}"
folder_path = os.path.join(base_dir, folder_name)
file_path = os.path.join(folder_path, "MSE_all_trans.npy")
# para = np.load( os.path.join(folder_path, "parameter_list.npy"), allow_pickle=True )
# =========================
# Load data
# =========================
A_sizes = [3, 5, 8, 10, 12, 15]
MSE_all = np.load(file_path)


titles = [
    'MSE for initial estimation',
    'MSE for debiased estimation',
    'MSE for dp-debiased estimation'
]

methods = ['Initial', 'Debiased', 'DP-Debiased']

# Create results directory if it doesn't exist
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Prepare data for plotting
data = []
for i, method in enumerate(methods):
    for j, a_size in enumerate(A_sizes):
        for k in range(MSE_all.shape[2]):
            data.append([MSE_all[i, j, k], method, a_size])

df = pd.DataFrame(data, columns=['MSE', 'Method', 'A_size'])

# Calculate mean and std deviation
summary = df.groupby(['Method', 'A_size'])['MSE'].agg(['mean', 'std']).reset_index()

# Map A_size to evenly spaced indices
a_size_indices = list(range(len(A_sizes)))
a_size_map = dict(zip(A_sizes, a_size_indices))
summary['A_index'] = summary['A_size'].map(a_size_map)

# Plot with evenly spaced x-axis
plt.figure(figsize=(12, 8))
for method in methods:
    sub_df = summary[summary['Method'] == method]
    plt.errorbar(
        x=sub_df['A_index'],
        y=sub_df['mean'],
        yerr=sub_df['std'],
        label=method,
        marker='o',
        capsize=5,
        linestyle='-',
    )


plt.xlabel(r'$|\mathcal{A}|$', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(title='Method', loc='upper right', fontsize=20)
plt.tight_layout()

save_name = f"MSE_n{n}n0{n0}p{p}.png"
save_path = os.path.join(folder_path, save_name)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)



