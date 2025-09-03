# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:49:18 2025

@author: user
"""

import pandas as pd;
import numpy as np;

from modelValidator_processor import ModelValidator;
from data_pre_processoer import data_pre_funs;

validator_funs = ModelValidator();
pre_funs = data_pre_funs();

df_train = pd.read_parquet("pre_datas_0822.parquet");


