# %%
import pandas as pd
import numpy as np
from util import MaxImputer, OutlierHandler, predict
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

import pytd.pandas_td as td
import warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=FutureWarning) 

# %%
MY_API_KEY = '581/8aa4544d308ca2ec6079d5dbeb8211cc1895b5f8'

con = td.connect(apikey=MY_API_KEY, endpoint='https://api.treasuredata.co.jp')
engine = td.create_engine("presto:ds_data_modeling", con=con)

# %%
# Config
SEED = 20250215
# Data
sd = '202401'
ed = '202407'
in_db_name = 'ds_data_modeling'
in_table_name = 'cli_mkt_data_for_modeling'
# Pipeline
pipeline_fn = 'models_prod/pipeline_cli_mkt.pkl'

# Output
out_col_name = 'cli_marketing_score'
out_db_name = 'ds_prediction'
out_table_name = 'cli_marketing_score'

# target_col = 'is_increase'
query_features = ['max_limit_cr_m4_m6', 'count_spending_per_month_m1_m6',
       'count_revolver_pmt_m1_m6', 'avg_limit_cr_m1_m3',
       'count_on_time_pmt_m1_m6', 'limit_cr', 'sum_amt_per_month_m1_m3',
       'days_from_last_sys_inc_limit_date',
       'min_amt_per_txn_sub_prod_type_cd_m1_m3', 'avg_limit_cr_m1_m6',
       'min_limit_cr_m1_m3', 'total_current_point', 'count_pmt_m1_m3',
       'sum_amt_per_month', 'days_from_last_salary_update',
       'min_amt_per_txn_sub_prod_type_cd',
       'days_from_last_sys_inc_limit_date_m4_m6', 'education',
       'min_amt_per_txn_sub_prod_type_cd_m1_m6',
       'min_no_day_bf_due_m1_m3', 'sum_amt_per_month_m4_m6',
       'avg_no_day_bf_due', 'max_pmt_amt_per_pmt_m1_m3',
       'max_amt_per_month_m4_m6', 'avg_pmt_amt_mthly_m1_m3',
       'sum_ol_amt_per_month_merchant_group_wellness']

# %%
# Load Pipeline
print(f'Load Pipeline from {pipeline_fn}')
grid = joblib.load(pipeline_fn)

# %%
# Load Table
query=f'''
    SELECT date_of_data, aeon_id,
        {','.join(query_features)}
    FROM {in_db_name}.{in_table_name}
    WHERE date_of_data >= '{sd}' AND date_of_data <= '{ed}'
        AND card_type_f = 'YC'
        AND consent_sts_nt0 = 1 AND consent_sts_dc0 = 1
'''

df = td.read_td_query(query, engine)
print(df.shape)

# %%
cat_cols = ['education']
num_cols = list(set(df.columns) - set(cat_cols + ['date_of_data']))

# %%
print('cat_cols',cat_cols)

# %%
print('num_cols',num_cols)

# %%
# Force NA
# Num
replace_cols = ['education']
print('replace_cols',replace_cols)
for c in replace_cols:
    df[c] = df[c].replace(99, np.nan)
    df.loc[df[c].notna(), c] = df.loc[df[c].notna(), c].astype(int)

# Force type of cat
df = df.astype({c : str for c in cat_cols})

# %%
# Prediction
y_score = predict(grid, df)

# %%
# Export
df_res = df[['aeon_id','date_of_data']]
df_res[out_col_name] = y_score
df_res = df_res.astype({'aeon_id':str, 'date_of_data':str, out_col_name:float})

# Write to table
td.to_td(df_res, f'{out_db_name}.{out_table_name}', con, if_exists='append', index=False)

# %%



