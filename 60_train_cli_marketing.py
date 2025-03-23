# %%
import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from util import MaxImputer, OutlierHandler, evaluate
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
ed = '202410'
in_db_name = 'ds_data_modeling'
in_table_name = 'cli_mkt_data_for_modeling'

# Output
pipeline_fn = 'models_prod/pipeline_cli_mkt.pkl'

target_col = 'is_increase'
query_features = [
'sum_ol_amt_per_month_merchant_group_wellness',
'count_spending_per_month_m1_m6', 'max_limit_cr_m4_m6',
'sum_amt_per_month_m1_m3', 'min_amt_per_txn_sub_prod_type_cd',
'sum_amt_per_month', 'min_amt_per_txn_sub_prod_type_cd_m1_m3',
'count_on_time_pmt_m1_m6', 'limit_cr', 'count_pmt_m1_m3',
'count_revolver_pmt_m1_m6', 'avg_limit_cr_m1_m3',
'total_current_point', 'max_pmt_amt_per_pmt_m1_m3', 'education',
'days_from_last_sys_inc_limit_date',
'min_amt_per_txn_sub_prod_type_cd_m1_m6', 'avg_limit_cr_m1_m6',
'avg_pmt_amt_mthly_m1_m3', 'min_limit_cr_m1_m3',
'sum_amt_per_month_m4_m6', 'days_from_last_salary_update',
'avg_no_day_bf_due', 'min_no_day_bf_due_m1_m3',
'max_amt_per_month_m4_m6',
'days_from_last_sys_inc_limit_date_m4_m6'
]

# %%
query=f'''
    SELECT {target_col},{','.join(query_features)}
    FROM {in_db_name}.{in_table_name}
    WHERE date_of_data >= '{sd}' AND date_of_data <= '{ed}'
        AND {target_col} is not null
        AND card_type_f = 'YC'
        AND consent_sts_nt0 = 1 AND consent_sts_dc0 = 1
'''

df = td.read_td_query(query, engine)
print('len data : ', df.shape)

# %%
print(df[target_col].value_counts(normalize=True))

# %% [markdown]
# ## 1. Cleansing

# %%
cat_cols = ['education']
num_cols = list(set(df.columns) - set(cat_cols + [target_col] + ['date_of_data']))

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

# %% [markdown]
# ### Fill NA

# %%
# Num
fill_mean = []
fill_max = [c for c in num_cols if 'days_from' in c]
fill_zero = list(set(num_cols) - set(fill_max) - set(fill_mean))
# Cat fill with '9999'=another bin

# %%
print('fill_mean',fill_mean)
print('fill_max',fill_max)
print('fill_zero',fill_zero)

# %% [markdown]
# ## 2. Data Modeling

# %% [markdown]
# ### Split

# %%
X_train = df.drop(columns=[target_col])
y_train = df[target_col]

print('X_train : ', X_train.shape)
print('y_train : ', y_train.shape)
del df #, df_train, df_test # When data is large

# %% [markdown]
# ### Imputer

# %%
imputer = DataFrameMapper(
                [([c], SimpleImputer(strategy='mean')) for c in fill_mean]
                + [([c], MaxImputer()) for c in fill_max]
                + [([c], SimpleImputer(strategy='constant', fill_value=0)) for c in fill_zero]
                + [([c], SimpleImputer(strategy='constant', fill_value='9999')) for c in cat_cols]
                , df_out=True
                )

# %% [markdown]
# ### Preprocessing

# %%
preprocessor_num = DataFrameMapper(
                [([c], [OutlierHandler(), StandardScaler()]) for c in num_cols]
                + [([c], None) for c in cat_cols]
                , df_out=True
                )
preprocessor_cat = DataFrameMapper(
                [([c], None) for c in num_cols]
                + [([c], OneHotEncoder(handle_unknown='value', use_cat_names=True)) for c in cat_cols]
                , df_out=True
                )

# %% [markdown]
# ## Modeling

# %%
model = {'name': 'xgboostclassifier',
           'model': XGBClassifier(),
           'param_grid': {
                    'xgboostclassifier__n_estimators': [100],
                    'xgboostclassifier__max_depth': [5],
                    'xgboostclassifier__random_state': [SEED]
                    
                } 
            }

# %%
def tune_model_hyperparams(X, y, model, selected_features=None, **common_grid_kwargs):
    print(f'{ model["name"] :-^70}')

    pipe_steps = None    
    # Check whether model is LR
    if model['name'] != 'logisticregression':
        # If not LR -> do feature selection first
        pipe_steps = [("imputer", imputer),
                      ("prepnum", preprocessor_num),
                      ("prepcat", preprocessor_cat)]

        if selected_features is not None:
            selector = DataFrameMapper([(col, None) for col in selected_features], df_out=True)
            pipe_steps += [('selector', selector)]
        else:
            pipe_steps += [('selector', RFE(estimator=model['model']))]
    else:
        # If LR -> no need (feature selection implemented by regularization)
        pipe_steps = [("imputer", imputer),
                      ("prepcat", preprocessor_cat)]

    pipe_steps += [(model['name'], model['model'])]

    grid = (GridSearchCV(Pipeline(pipe_steps),
                                         param_grid=model['param_grid'],
                                         **common_grid_kwargs)
                                .fit(X, y))

    return grid

def print_grid_results(grids):
    for name, model in grids.items():
        print(f'{name :-^70}')
        print(f'Score:\t\t{model.best_score_:.2%}')
        print(f'Parameters:\t{model.best_params_}')
        print('*' * 70)

# %%
# Change HERE
selected_features = [
'sum_ol_amt_per_month_merchant_group_wellness', # add
'count_spending_per_month_m1_m6', 'max_limit_cr_m4_m6', 'sum_amt_per_month_m1_m3', 'min_amt_per_txn_sub_prod_type_cd', 'sum_amt_per_month', 'min_amt_per_txn_sub_prod_type_cd_m1_m3', 'count_on_time_pmt_m1_m6', 'limit_cr', 'count_pmt_m1_m3', 'count_revolver_pmt_m1_m6', 'avg_limit_cr_m1_m3', 'total_current_point', 'max_pmt_amt_per_pmt_m1_m3', 'education_0_4.0', 'days_from_last_sys_inc_limit_date', 'min_amt_per_txn_sub_prod_type_cd_m1_m6', 'avg_limit_cr_m1_m6', 'avg_pmt_amt_mthly_m1_m3', 'min_limit_cr_m1_m3', 'sum_amt_per_month_m4_m6', 'days_from_last_salary_update', 'avg_no_day_bf_due', 'min_no_day_bf_due_m1_m3', 'max_amt_per_month_m4_m6', 'days_from_last_sys_inc_limit_date_m4_m6'
]

# %%
# Change HERE
st = time.time()
np.random.seed(SEED)

# Training
scoring = ['roc_auc','f1','precision','recall']

grid = tune_model_hyperparams(X_train, y_train, model, selected_features=selected_features,
                               verbose=1, n_jobs=1,
                               scoring=scoring, refit='roc_auc', cv=5, return_train_score=True)
    
et = time.time()
print(f'Time Train : {(et-st)/60} mins.')
# 8cores 20mins/XGgrid -> total 3hrs

# %%
# Save
joblib.dump(grid, pipeline_fn)
print(f'Pipeline saved as {pipeline_fn}')

# %% [markdown]
# ## 3. Evaluation

# %%
# Predict with train set
# %1 = 60.6
# mycutoff = 1-y_train.mean()
result_train, y_score, mycutoff = evaluate(grid, y_train, X_train, cutoff=None)
print(pd.DataFrame({k: [v] for k, v in result_train.items()}).T)

# %%
print("Cutoff", mycutoff)


