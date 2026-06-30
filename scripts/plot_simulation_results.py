# %%
paras = 'n4000n04000p200corr02'
test_path = './code/results/ex1_3/'+ paras +'/'
############################################
# Simulation on whole transfer process
############################################

# %%
paras = 'test'
test_path = './code/results/ex1_1/'+ paras +'/'
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# A_sizes = [3, 5, 8, 10, 12, 15]
# MSE_all = np.load(test_path+'MSE_all_trans.npy')
# # Assuming MSE_all and A_sizes are already defined

# titles = [
#     'MSE for initial estimation',
#     'MSE for debiased estimation',
#     'MSE for dp-debiased estimation'
# ]

# methods = ['Initial', 'Debiased', 'DP-Debiased']

# # Create the results directory if it doesn't exist
# results_dir = './results'
# os.makedirs(results_dir, exist_ok=True)

# # Reshape data for seaborn
# data = []
# for i, method in enumerate(methods):
#     for j, a_size in enumerate(A_sizes):
#         for k in range(MSE_all.shape[2]):
#             data.append([MSE_all[i, j, k], method, a_size])

# df = pd.DataFrame(data, columns=['MSE', 'Method', 'A_size'])

# # Calculate mean MSE for each method and A_size
# mean_MSE = df.groupby(['Method', 'A_size'])['MSE'].mean().reset_index()

# # Boxplot for all methods grouped by A_size
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='A_size', y='MSE', hue='Method', data=df)

# # Overlay mean MSE
# for i, method in enumerate(methods):
#     means = mean_MSE[mean_MSE['Method'] == method]
#     plt.plot(np.arange(len(A_sizes)), means['MSE'], marker='o', linestyle='-', label=f'{method} Mean')

# plt.title('MSE Comparison for Different Methods and A_sizes')
# plt.xlabel('A_size')
# plt.ylabel('MSE')
# plt.xticks(ticks=np.arange(len(A_sizes)), labels=A_sizes, rotation=45)
# plt.legend(title='Method', loc='upper right')
# plt.tight_layout()

# # Save the combined plot
# combined_plot_filename = os.path.join(results_dir, 'Combined_MSE_Comparison_with_Mean.png')
# plt.savefig(combined_plot_filename)
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

A_sizes = [3, 5, 8, 10, 12, 15]
MSE_all = np.load(test_path + 'MSE_all_trans.npy')

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

plt.title(r'MSE Comparison for Different Methods and $|\mathcal{A}|$', fontsize=20)
plt.xlabel(r'$|\mathcal{A}|$', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
plt.legend(title='Method', loc='upper right', fontsize=20)
plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# ncol = len(labels)

# plt.legend(
#     handles=handles, 
#     labels=labels, 
#     title='Method', 
#     loc='lower center', 
#     bbox_to_anchor=(0.5, -0.35),  # Position below the plot
#     ncol=ncol,  # Number of columns = number of entries
#     fontsize=15
# )
# plt.tight_layout()
# Save plot
plot_filename = os.path.join(results_dir, ('MSE_'+paras+'.png'))
plt.savefig(plot_filename)
plt.show()


# %%
# import numpy as np
# Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
# Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
# A_sizes = [3, 5, 8, 10, 12, 15]
# # A_sizes = [12, 15, 20]
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# # Plot the histogram of Bias_de and Bias_de_dp for different sample sizes
# for i, n in enumerate(A_sizes):
#     plt.figure(figsize=(15, 6))
#     plt.subplot(121)
#     plt.hist(1*Bias_de_all[i][0,np.abs(Bias_de_all[i][0,:])<40], bins=20, alpha=0.75, density=True, label=f'Bias_de (n={n})')
#     x = np.linspace(-4, 4, 100)
#     plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
#     plt.xlabel('Bias')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of Bias_de for source number={n}')
#     # plt.xlim(-4, 4)  # Set x-axis limits
#     plt.legend()
    
#     plt.subplot(122)
#     plt.hist(Bias_de_dp_all[i][0,np.abs(Bias_de_all[i][0,:])<40], bins=20, alpha=0.75, density=True, label=f'Bias_de_dp (n={n})')
#     x = np.linspace(-4, 4, 100)
#     plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
#     plt.xlabel('Bias')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of Bias_de_dp for source number={n}')
#     plt.xlim(-4, 4)  # Set x-axis limits
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(f'bias_histogram_n_{n}.png')
#     plt.show()
paras = 'test'
test_path = './code/results/ex1_1/'+ paras +'/'
import scipy.stats as stats
Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
# A_sizes = [3, 5, 8, 10, 12, 15]
A_sizes = [5, 10, 15]
fig, axes = plt.subplots(len(A_sizes), 3, figsize=(25, 6 * len(A_sizes)))

for i, n in enumerate(A_sizes):
    # Plot Bias_de histogram
    ax1 = axes[i, 0]
    ax1.hist(Bias_de_dp_all[i][0, :], bins=20, alpha=0.75, density=True, label='Bias')
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
    ax1.set_xlabel('Bias', fontsize=20)
    ax1.set_ylabel('Frequency', fontsize=20)
    # ax1.set_title(r'Histogram of Bias for $\hat{\beta}_{1}^{(dp)}$', fontsize=20)  # Increased title size
    ax1.set_title(r'$\hat{\beta}_{1}^{(dp)}$', fontsize=20) 
#     ax1.legend(fontsize=15) 

    # Plot Bias_de_dp histogram
    ax2 = axes[i, 1]
    ax2.hist(Bias_de_dp_all[i][14, :], bins=20, alpha=0.75, density=True, label='Bias')
    ax2.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
    ax2.set_xlabel('Bias', fontsize=20)
    ax2.set_ylabel('Frequency', fontsize=20)
    # ax2.set_title(r'Histogram of Bias for $\hat{\beta}_{15}^{(dp)}$', fontsize=20)
    ax2.set_title(r'$\hat{\beta}_{15}^{(dp)}$', fontsize=20)
    ax2.set_xlim(-4, 4)
    # ax2.legend(fontsize=15)

    ax3 = axes[i, 2]
    ax3.hist(Bias_de_dp_all[i][29, :], bins=20, alpha=0.75, density=True, label='Bias')
    ax3.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
    ax3.set_xlabel('Bias', fontsize=20)
    ax3.set_ylabel('Frequency', fontsize=20)
    ax3.set_title(r'$\hat{\beta}_{30}^{(dp)}$', fontsize=20)
    ax3.set_xlim(-4, 4)
    # ax3.legend(fontsize=15)

plt.tight_layout()
plt.savefig('./code/fig/hist/' + paras + '_combined_bias_histograms.png')
plt.show()

#%%
# plt.figure(figsize=(15, 9))
# plt.hist(Bias_de_dp_all[i][0, :], bins=20, alpha=0.75, density=True, label='Bias')
# x = np.linspace(-4, 4, 100)
# plt.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal')
# plt.xlabel('Bias', fontsize=20)
# plt.ylabel('Frequency', fontsize=20)
# plt.title(r'Histogram of Bias for $\hat{\beta}_{1}^{(dp)}$', fontsize=20) 
# handles, labels = plt.gca().get_legend_handles_labels()
# ncol = len(labels)

# plt.legend(
#     handles=handles, 
#     labels=labels, 
#     title='Method', 
#     loc='lower center', 
#     bbox_to_anchor=(0.5, -0.35),  # Position below the plot
#     ncol=ncol,  # Number of columns = number of entries
#     fontsize=15
# )
# plt.tight_layout()
# %%
import numpy as np
import pandas as pd
paras = 'test'
test_path = './code/results/ex1_1/'+ paras +'/'
A_sizes = [3, 5, 8, 10, 12, 15]
# A_sizes = [12, 15, 20]
S=(np.concatenate([np.ones(10), np.zeros(990)])>0)
# S[4] = 1
IR_all = np.load(test_path+'IR_all_trans.npy')
MSE_all = np.load(test_path+'MSE_all_trans.npy')
cilen_all = np.load(test_path+'cilen_all_trans.npy')
beta_true_all = np.load(test_path+'beta_true_all_trans.npy')
Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
# Assuming A_sizes, MSE_all, cilen_all, IR_all, and beta_true_all are defined

# Create lists to hold the output values
A_size_list = []
mean_MSE_list = []
std_MSE_list = []
mean_cilen_list = []
mean_IR_S_list = []
mean_IR_not_S_list = []
mean_IR_all_list = []

for i, n in enumerate(A_sizes):
    A_size_list.append(n)
    mean_MSE_list.append(np.mean(MSE_all[2, i]))
    std_MSE_list.append(np.std(MSE_all[2, i]))
    mean_cilen_list.append(np.mean(cilen_all[i]))
    
    # S = ~(beta_true_all[i] == 0)
    mean_IR_S_list.append(np.mean(np.mean(IR_all[i], axis=1)[S]))
    mean_IR_not_S_list.append(np.mean(np.mean(IR_all[i], axis=1)[~S]))
    mean_IR_all_list.append(np.mean(np.mean(IR_all[i], axis=1)))

# Create a DataFrame to hold the results
result_df = pd.DataFrame({
    'A_size': A_size_list,
    'Mean_MSE': mean_MSE_list,
    'Std_MSE': std_MSE_list,
    'Mean_CILEN': mean_cilen_list,
    'Mean_IR_S': mean_IR_S_list,
    'Mean_IR_not_S': mean_IR_not_S_list,
    'Mean_IR_All': mean_IR_all_list
})

# Print the table
# print(result_df)
result_df


# %%
IR_all = np.load(test_path+'IR_all_trans.npy')
# IR_all
plt.plot(np.mean(IR_all[5], axis=1))

# %%
fig, axes = plt.subplots(len(A_sizes), 1, figsize=(15, 6 * len(A_sizes)))

for i, n in enumerate(A_sizes):
    ax1 = axes[i]
    ax1.plot(np.mean(IR_all[i], axis=1), alpha=0.75)


# %%
for i, n in enumerate(A_sizes):
    print(n)
    print(np.mean(MSE_all[2,i]))
    print(np.std(MSE_all[2,i]))
    print(np.mean(cilen_all[i]))
    # print(np.mean(IR_all[i], axis=1))
    S = ~(beta_true_all[i] == 0)
    print(np.mean(np.mean(IR_all[i], axis=1)[S]))
    print(np.mean(np.mean(IR_all[i], axis=1)[~S]))
    print(np.mean(np.mean(IR_all[i], axis=1)))


# %%
############################################
# Simulation on MSE with no transfer
############################################
paras = 'test'
test_path = './code/results/ex1_1/'+ paras +'/'

# %%
import numpy as np
import pandas as pd
A_sizes = [3, 5, 8, 10, 12, 15]
S=(np.concatenate([np.ones(10), np.zeros(990)])>0)
IR_all = np.load(test_path+'IR_all_trans.npy')
MSE_all = np.load(test_path+'MSE_all_trans.npy')
cilen_all = np.load(test_path+'cilen_all_trans.npy')
beta_true_all = np.load(test_path+'beta_true_all_trans.npy')
Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
# Assuming A_sizes, MSE_all, cilen_all, IR_all, and beta_true_all are defined

# Create lists to hold the output values
A_size_list = []
mean_MSE_list = []
std_MSE_list = []
mean_cilen_list = []
mean_IR_S_list = []
mean_IR_not_S_list = []
mean_IR_all_list = []

for i, n in enumerate(A_sizes):
    A_size_list.append(n)
    mean_MSE_list.append(np.mean(MSE_all[2, i]))
    std_MSE_list.append(np.std(MSE_all[2, i]))
    mean_cilen_list.append(np.mean(cilen_all[i]))
    
    # S = ~(beta_true_all[i] == 0)
    mean_IR_S_list.append(np.mean(np.mean(IR_all[i], axis=1)[S]))
    mean_IR_not_S_list.append(np.mean(np.mean(IR_all[i], axis=1)[~S]))
    mean_IR_all_list.append(np.mean(np.mean(IR_all[i], axis=1)))

# Create a DataFrame to hold the results
result_df = pd.DataFrame({
    'A_size': A_size_list,
    'Mean_MSE': mean_MSE_list,
    'Std_MSE': std_MSE_list,
    'Mean_CILEN': mean_cilen_list,
    'Mean_IR_S': mean_IR_S_list,
    'Mean_IR_not_S': mean_IR_not_S_list,
    'Mean_IR_All': mean_IR_all_list
})

# Print the table
# print(result_df)
result_df

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# A_sizes = [3, 5, 8, 10, 12, 15]
A_sizes = [3, 5, 8, 10, 12]
S=(np.concatenate([np.ones(10), np.zeros(990)])>0)
IR_all = np.load(test_path+'IR_all_trans.npy')
MSE_all = np.load(test_path+'MSE_all_trans.npy')

# Create lists to hold the output values
A_size_list = []
mean_MSE_init_list = []
mean_MSE_de_list = []
mean_MSE_dedp_list = []
mean_MSE_targ_list = []
mean_MSE_nosy_list = []
std_MSE_init_list = []
std_MSE_de_list = []
std_MSE_dedp_list = []
std_MSE_targ_list = []
std_MSE_nosy_list = []

for i, n in enumerate(A_sizes):
    A_size_list.append(n)
    mean_MSE_init_list.append(np.mean(MSE_all[0, i]))
    std_MSE_init_list.append(np.std(MSE_all[0, i]))
    mean_MSE_de_list.append(np.mean(MSE_all[1, i]))
    std_MSE_de_list.append(np.std(MSE_all[1, i]))
    mean_MSE_dedp_list.append(np.mean(MSE_all[2, i]))
    std_MSE_dedp_list.append(np.std(MSE_all[2, i]))
    # mean_MSE_targ_list.append(np.mean(MSE_all[3, i]))
    # std_MSE_targ_list.append(np.std(MSE_all[3, i]))
    # mean_MSE_nosy_list.append(np.mean(MSE_all[4, i]))
    # std_MSE_nosy_list.append(np.std(MSE_all[4, i]))

# Create a DataFrame to hold the results
result_df = pd.DataFrame({
    'A_size': A_size_list,
    'Mean_MSE_init': mean_MSE_init_list,
    'Std_MSE_init': std_MSE_init_list,
    'Mean_MSE_de': mean_MSE_de_list,
    'Std_MSE_de': std_MSE_de_list,
    'Mean_MSE_dedp': mean_MSE_dedp_list,
    'Std_MSE_dedp': std_MSE_dedp_list,
    # 'Mean_MSE_targ': mean_MSE_targ_list,
    # 'Std_MSE_targ': std_MSE_targ_list,
    # 'Mean_MSE_nosy': mean_MSE_nosy_list,
    # 'Std_MSE_nosy': std_MSE_nosy_list
})

# Print the table
# print(result_df)
result_df
plt.figure(figsize=(10, 6))
plt.plot(A_size_list, mean_MSE_init_list, marker='o', alpha=0.5)
plt.plot(A_size_list, mean_MSE_de_list, marker='o', alpha=0.5)
plt.plot(A_size_list, mean_MSE_dedp_list, marker='o', alpha=0.5)
# plt.plot(A_size_list, mean_MSE_targ_list, marker='o', alpha=0.5)
# plt.plot(A_size_list, mean_MSE_nosy_list, marker='o', alpha=0.5)
plt.legend(['Mean_MSE_init', 'Mean_MSE_de', 'Mean_MSE_dedp', 'Mean_MSE_targ', 'Mean_MSE_nosy'])
plt.title('Mean MSE for Different Methods')
plt.xlabel('A_size')
plt.ylabel('Mean MSE')
plt.show()

# %%
# Create a box plot for all MSE
# plt.figure(figsize=(12, 8))
# data = [MSE_all[j, i] for i in range(len(A_sizes)) for j in range(5)]
# labels = [f'{method} (n={n})' for n in A_sizes for method in ['MSE_init', 'MSE_de', 'MSE_dedp', 'MSE_targ', 'MSE_nosy']]
# plt.boxplot(data, labels=labels)
# plt.xticks(rotation=90)
# plt.title('Box Plot of MSE for Different Methods and A_sizes')
# plt.xlabel('Method and A_size')
# plt.ylabel('MSE')
# plt.show()


# %%
for i, n in enumerate(A_sizes):
    print(n)
    print(np.mean(MSE_all[2,i]))
    print(np.std(MSE_all[2,i]))
    print(np.mean(cilen_all[i]))
    # print(np.mean(IR_all[i], axis=1))
    S = ~(beta_true_all[i] == 0)
    print(np.mean(np.mean(IR_all[i], axis=1)[S]))
    print(np.mean(np.mean(IR_all[i], axis=1)[~S]))
    print(np.mean(np.mean(IR_all[i], axis=1)))

# %%
############################################
# Simulation on e-BH
############################################

# paras = 'test'
paras = 'n=1000,n0=500,p=500'
test_path = './results/'+ paras +'/'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
A_sizes = [3, 5, 8, 10, 12, 15]

# %%
def peeling(v, s, epsilon, delta, lambda_):
    d = len(v)
    S = []
    # print(s)
    # s = int(s)
    
    # Generate noise for the initial selection process
    for j in range(s):
        noise = np.random.laplace(0, 2 * lambda_ * np.sqrt(3 * s * np.log(1 / delta)) / epsilon, d)
        available_indices = list(set(range(d)) - set(S))
        noisy_values = np.abs(v[available_indices]) + noise[available_indices]
        j_star = available_indices[np.argmax(noisy_values)]
        S.append(j_star)
    
    # Generate noise for the output vector
    w = np.random.laplace(0, 2 * lambda_ * np.sqrt(3 * s * np.log(1 / delta)) / epsilon, d)
    # w = 0
    # Construct the output vector
    v_tilde = np.zeros(d)
    for j in S:
        v_tilde[j] = v[j] + w[j]
    
    return v_tilde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# e_noisy_value_all = np.load(test_path+"e_noisy_values.npy")
e_value_all = np.load(test_path+"e_values.npy")
p_value_all = np.load(test_path+"p_values.npy")
beta_true_all = np.load(test_path+"beta_true_all_trans.npy")
# para_list = np.load(test_path+"parameter_list.npy", allow_pickle=True)
results_dir = './results'

Nsim = 500#para_list.item(0)['Nsim']
epsilon = 0.6#para_list.item(0)['epsilon']
delta = (1/1000)**1.1#para_list.item(0)['delta']
# 假设 A_sizes, beta_true_all, e_value_all, p_value_all 这些变量已经定义
data = []
noise_scale = 0.2

for i, n in enumerate(A_sizes):
    beta_true = beta_true_all[i]
    # e_noisy_values = np.array(e_noisy_value_all[i])
    e_values = np.array(e_value_all[i]) * 2
    # e_values = np.mean(np.array(e_value_all[i]), axis=1)
    # p_noisy_values = np.array(e_noisy_value_all[i])
    p_values = np.array(p_value_all[i])  + 1e-10
    # p_values = np.mean(np.array(p_value_all[i]), axis=1)
    d = len(e_values)
    e_noisy_star = np.zeros(shape=e_values.shape)
    e_star = np.zeros(shape=e_values.shape)
    p_noisy_star = np.zeros(shape=p_values.shape)
    p_star = np.zeros(shape=p_values.shape)
    # for l in range(Nsim):
    #     p_noisy_values[:,l] = peeling(np.log(p_values[:,l]), d, epsilon / 2, delta / 2, noise_scale)
    #     p_noisy_star[:,l] = peeling(np.log(p_star[:,l]), d, epsilon / 2, delta / 2, noise_scale)

    FDP_e_noisy = np.zeros(e_values.shape[1])
    FDP_e = np.zeros(e_values.shape[1])
    FDP_p_noisy = np.zeros(e_values.shape[1])
    FDP_p = np.zeros(e_values.shape[1])
    Size_e_noisy = np.zeros(e_values.shape[1])
    Size_e = np.zeros(e_values.shape[1])
    Size_p_noisy = np.zeros(e_values.shape[1])
    Size_p = np.zeros(e_values.shape[1])
    for k in range(e_values.shape[1]):
        # e_star[:,k] = -(np.linspace(1, d, d) * np.sort(-np.array(e_values)[:,k])) / d
        # e_noisy_values[:,k] = peeling((e_values[:,k]), d, epsilon / 2, delta / 2, noise_scale)
        # e_noisy_star[:,k] = -(np.linspace(1, d, d) * np.sort(-np.sum(np.array(e_noisy_values)[:,k:k+1], axis=1))) / d
        # e_star[:,k] = -2 * (np.linspace(1, d, d) * np.sort(-np.sum(np.array(e_values)[:,k:(k+1)], axis=1))) / d
        e_star[:,k] = -1 * (np.linspace(1, d, d) * np.sort(-np.array(e_values)[:,k])) / d
        # p_noisy_values[:,k] = peeling(np.log(p_values[:,k]), d, epsilon / 2, delta / 2, noise_scale)
        p_star[:,k] = (np.linspace(1, d, d) / np.sort(np.array(p_values)[:,k])) / d
        p_noisy_star[:,k] = peeling(np.log(p_star[:,k]), d, epsilon / 2, delta / 2, noise_scale)
    
        FDP_e_noisy[k] = np.sum((e_noisy_star[:,k] > 20) * (beta_true == 0)) / max(np.sum(e_noisy_star[:,k] > 20), 1)
        FDP_e[k] = np.sum((e_star[:,k] > 20) * (beta_true == 0)) / max(np.sum(e_star[:,k] > 20), 1)
        FDP_p_noisy[k] = np.sum((p_noisy_star[:,k] > np.log(20)) * (beta_true == 0)) / max(np.sum(p_star[:,k] > np.log(20)), 1)
        FDP_p[k] = np.sum((p_star[:,k] > 20) * (beta_true == 0)) / max(np.sum(p_star[:,k] > 20), 1)
    
        Size_e_noisy[k] = np.sum((e_noisy_star[:,k] > 20) * (beta_true != 0)) / np.sum(beta_true != 0)
        Size_e[k] = np.sum((e_star[:,k] > 20) * (beta_true != 0)) / np.sum(beta_true != 0)
        Size_p_noisy[k] = np.sum((p_noisy_star[:,k] > np.log(20)) * (beta_true != 0)) / np.sum(beta_true != 0)
        Size_p[k] = np.sum((p_star[:,k] > 20) * (beta_true != 0)) / np.sum(beta_true != 0)
    
    FDR_e_noisy = np.mean(FDP_e_noisy)
    FDR_e = np.mean(FDP_e)
    FDR_p_noisy = np.mean(FDP_p_noisy)
    FDR_p = np.mean(FDP_p)
    Power_e_noisy = np.mean(Size_e_noisy)
    Power_e = np.mean(Size_e)
    Power_p_noisy = np.mean(Size_p_noisy)
    Power_p = np.mean(Size_p)
    # 将结果添加到表格数据中
    data.append([n, FDR_e_noisy, FDR_e, FDR_p_noisy, FDR_p, Power_e_noisy, Power_e, Power_p_noisy, Power_p])

# 使用pandas将数据转换为DataFrame
df = pd.DataFrame(data, columns=['A_size', 'FDR_e_noisy', 'FDR_e', 'FDR_p_noisy', 'FDR_p', 'Power_e_noisy', 'Power_e', 'Power_p_noisy', 'Power_p'])

# 显示表格
print(df)


# 绘制FDP的折线图
plt.figure(figsize=(12, 8))

a_size_indices = range(len(A_sizes))
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
plt.plot(a_size_indices, df['FDR_e'], label='FDR_e', marker='o', color='blue')
plt.plot(a_size_indices, df['FDR_p'], label='FDR_p', marker='s', color='orange')

# 添加标题和轴标签
# plt.title('FDR vs A_size', pad=20)
plt.xlabel(r'$|\mathcal{A}|$', fontsize=20)
plt.ylabel('FDR', fontsize=20)

# 添加图例，设置位置和标题
# plt.legend(loc='upper right', fontsize=35)
# handles, labels = plt.gca().get_legend_handles_labels()
# ncol = len(labels)

# plt.legend(
#     handles=handles, 
#     labels=labels, 
#     title='Method', 
#     loc='lower center', 
#     bbox_to_anchor=(0.5, -0.35),  # Position below the plot
#     ncol=ncol,  # Number of columns = number of entries
#     fontsize=15
# )
# plt.tight_layout()

# 自动调整布局，避免标签或标题被截断
plt.tight_layout()

plot_filename = os.path.join(results_dir, ('FDR_'+paras+'.png'))
plt.savefig(plot_filename)
plt.show()

plt.figure(figsize=(12, 8))

a_size_indices = range(len(A_sizes))
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
plt.plot(a_size_indices, df['Power_e'], label='Power_e', marker='o', color='blue')
plt.plot(a_size_indices, df['Power_p'], label='Power_p', marker='s', color='orange')

# 添加标题和轴标签
# plt.title('FDR vs A_size', pad=20)
plt.xlabel('A_size', fontsize=20)
plt.ylabel('Power', fontsize=20)

# 添加图例，设置位置和标题
plt.legend(loc='upper right', fontsize=15)


# 自动调整布局，避免标签或标题被截断
plt.tight_layout()

plot_filename = os.path.join(results_dir, 'POWER_'+paras+'.png')
plt.savefig(plot_filename)
plt.show()

############################################
# Parameters
############################################
# %%
# para_list.item(0)['n']
Nsim = para_list.item(0)['Nsim']
epsilon = para_list.item(0)['epsilon']
delta = para_list.item(0)['delta']
# s = para_list.item(0)['delta']
p_value_noisy = np.zeros([d,Nsim])
p_star_noisy = np.zeros([d,Nsim])
noise_scale = 1e-2

# %%
############################################
# With no-noise
############################################

# paras = 'test'
paras = 'n=1000,n0=1000,w=02'
test_path = './results/'+ paras +'/'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

Bias_de_all = np.load(test_path+'Bias_de_all_trans.npy')
Bias_de_dp_all = np.load(test_path+'Bias_de_dp_all_trans.npy')
A_sizes = [3, 5, 8, 10, 12, 15]

MSE_all = np.load(test_path + 'MSE_all_trans.npy')
MSE_all_nonoise = np.load(test_path + 'MSE_all_trans_nonoise.npy')

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

# ----- Prepare non-noise data -----
data_nonoise = []
for j, a_size in enumerate(A_sizes):
    for k in range(MSE_all_nonoise.shape[2]):   # 500 reps
        data_nonoise.append([MSE_all_nonoise[0, j, k], a_size])

df_nonoise = pd.DataFrame(data_nonoise, columns=['MSE', 'A_size'])

summary_nonoise = (
    df_nonoise
    .groupby('A_size')['MSE']
    .agg(['mean', 'std'])
    .reset_index()
)
summary_nonoise['A_index'] = summary_nonoise['A_size'].map(a_size_map)

plt.errorbar(
    x=summary_nonoise['A_index'],
    y=summary_nonoise['mean'],
    yerr=summary_nonoise['std'],
    label='ZNE',
    marker='o',
    capsize=5,
    linestyle='-'
)

# plt.title(r'MSE Comparison for Different Methods and $|\mathcal{A}|$', fontsize=20)
plt.xlabel(r'$|\mathcal{A}|$', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(ticks=a_size_indices, labels=A_sizes, fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(title='Method', loc='upper right', fontsize=20)
# plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
ncol = len(labels)

plt.legend(
    handles=handles, 
    labels=labels, 
    title='Method', 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.35),  # Position below the plot
    ncol=ncol,  # Number of columns = number of entries
    fontsize=15
)
plt.tight_layout()
# Save plot
plot_filename = os.path.join(results_dir, ('MSE_'+paras+'.png'))
plt.savefig(plot_filename)
plt.show()

# %%

