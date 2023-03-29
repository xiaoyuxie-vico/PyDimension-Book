#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1.6: Sensitivity analysis

# - **Authors**: Xiaoyu Xie
# - **Contact**: xiaoyuxie2020@u.northwestern.edu
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiaoyuxie-vico/PyDimension-Book/blob/main/examples/sensitive_analysis.ipynb)
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xiaoyuxie-vico/PyDimension-Book/HEAD)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from SALib.sample import saltelli
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams["font.family"] = 'Arial'


# In[2]:


# # please uncomment these two lines, if you run this code in Colab
# !git clone https://github.com/xiaoyuxie-vico/PyDimension-Book
# %cd PyDimension-Book/examples


# ## Helper functions

# In[3]:


def parse_data(df, para_list, output='e*'):
    '''Parse the input and output parameters'''
    X = df[para_list].to_numpy()
    y = df[output].to_numpy()
    return X, y

def calculate_bounds(df, para_list):
    '''Calculate lower and upper bounds for each parameter'''
    bounds = []
    for var_name in para_list:
        bounds.append([df[var_name].min(), df[var_name].max()])
    return bounds

def train_model(X, y, coef_pi, deg):
    '''Build a predictive model with polynomial function'''
    # build features
    pi1 = np.prod(np.power(X, coef_pi.reshape(-1,)), axis=1).reshape(-1, 1)
    poly = PolynomialFeatures(deg)
    pi1_poly = poly.fit_transform(pi1)
    
    # fit
    model = LinearRegression(fit_intercept=False)
    model.fit(pi1_poly, y)
    model.score(pi1_poly, y)
    return model, poly

def SA(para_list, coef_pi, bounds, model, poly, sample_num=2**10):
    '''Sensitivity analysis'''
    problem = {'num_vars': len(para_list), 'names': para_list, 'bounds': bounds}

    # Generate samples
    X_sampled = saltelli.sample(problem, sample_num, calc_second_order=True)
    pi1_sampled = np.prod(np.power(X_sampled, coef_pi.reshape(-1,)), axis=1).reshape(-1, 1)
    pi1_sampled_poly = poly.transform(pi1_sampled)
    Y_sampled = model.predict(pi1_sampled_poly).reshape(-1,)
    print(Y_sampled.shape)

    # Perform analysis
    Si = sobol.analyze(problem, Y_sampled, print_to_console=True)
    
    return Si

def plot(Si, xtick_labels):
    '''Visualization'''
    total_Si, first_Si, second_Si = Si.to_df()
    
    total_Si['Type'] = ['Sobol total'] * total_Si.shape[0]
    total_Si = total_Si.rename(columns={'ST': 'Sensitivity', 'ST_conf': 'conf'})
    total_Si.index.name = 'Variable'
    total_Si.reset_index(inplace=True)

    first_Si['Type'] = ['Sobol 1st order'] * first_Si.shape[0]
    first_Si = first_Si.rename(columns={'S1': 'Sensitivity', 'S1_conf': 'conf'})
    first_Si.index.name = 'Variable'
    first_Si.reset_index(inplace=True)
    
    res_df = pd.concat([first_Si, total_Si]).reset_index(drop=False)
    # res_df = res_df.reindex(combined_df_index)
    
    fig = plt.figure()
    ax = sns.barplot(data=res_df, x='Variable', y='Sensitivity', hue='Type')
    ax.set_xticklabels(xtick_labels)
    ax.legend(fontsize=14)
    ax.set_xlabel('Variable', fontsize=16)
    ax.set_ylabel('Sensitivity', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()


# ## Load keyhole dataset

# In[4]:


# load data
df = pd.read_csv('../dataset/dataset_keyhole.csv')
df.describe()


# ## Analysis for Ke

# In[5]:


# config
para_list = ['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0']
coef_pi = np.array([1, -0.5, -1.5, -0.5, -1, -1, -1])  # for Ke
deg = 3

# choose parameters
X, y = parse_data(df, para_list)

# calculate bounds
bounds = calculate_bounds(df, para_list)

# train mdoel
model, poly = train_model(X, y, coef_pi, deg)

# calculate sensitivity
Si = SA(para_list, coef_pi, bounds, model, poly)


# In[6]:


xtick_labels = [r'$\eta P$', r'$\rho$', r'$T_l-T_0$', r'$\alpha$', r'$r_0$', r'$V_s$', r'$C_p$']
# sort the sensitivity from high to low
# combined_df_index = [0, 4, 6, 3, 2, 1, 5, 0+7, 4+7, 6+7, 3+7, 2+7, 1+7, 5+7] 
plot(Si, xtick_labels)


# ## Add one more parameter $T_v-T_l$

# In[7]:


# config
para_list = ['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0', 'Tv-T0']
coef_pi = np.array([1, -0.5, -1.5, -0.5, -1, -1, -0.75, -0.25])  # for table 3, 2nd row
deg = 3

# choose parameters
X, y = parse_data(df, para_list)

# calculate bounds
bounds = calculate_bounds(df, para_list)

# train mdoel
model, poly = train_model(X, y, coef_pi, deg)

# calculate sensitivity
Si = SA(para_list, coef_pi, bounds, model, poly)


# In[8]:


xtick_labels = [r'$\eta P$', r'$V_s$', r'$r_0$', r'$\alpha$', r'$\rho$',
                    r'$C_p$', r'$T_l-T_0$', r'$T_v-T_l$']
plot(Si, xtick_labels)


# ## Add one more parameter $L_m$

# In[9]:


# config
para_list = ['etaP', 'Vs', 'r0', 'alpha', 'rho', 'cp', 'Tl-T0', 'Lm']
coef_pi = np.array([1, -0.5, -1.5, -0.5, -1, -0.75, -0.75, -0.25])  # for table 4, 3rd row
deg = 3

# choose parameters
X, y = parse_data(df, para_list)

# calculate bounds
bounds = calculate_bounds(df, para_list)

# train mdoel
model, poly = train_model(X, y, coef_pi, deg)

# calculate sensitivity
Si = SA(para_list, coef_pi, bounds, model, poly)


# In[10]:


xtick_labels = [r'$\eta P$', r'$V_s$', r'$r_0$', r'$\alpha$', r'$\rho$',
                    r'$C_p$', r'$T_l-T_0$', r'$L_m$']
plot(Si, xtick_labels)


# In[ ]:




