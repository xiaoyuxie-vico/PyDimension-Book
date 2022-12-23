#!/usr/bin/env python
# coding: utf-8

# # Discover the governing equation from spring-mass-damper systems!

# In[1]:


import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.family"] = "Arial"
np.set_printoptions(suppress=True)


# In[ ]:




