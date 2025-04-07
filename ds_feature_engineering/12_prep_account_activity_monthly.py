import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import pytd.pandas_td as td

def run(DOD):
    eom_dt = (datetime.strptime(DOD, "%Y%m") + pd.offsets.MonthEnd(0))
    EOM = eom_dt.strftime("%Y%m%d")

    st = time.time()

    # %% [markdown]
    # ### 1. Change Limit Log

    # %%
    # payment due date at 10th
    query=f'''SELECT * FROM cr_change_limit_log
                WHERE SUBSTRING(input_date, 1, 6) = '{DOD}'
            '''

    df = td.read_td_query(query, engine)

    # %%
    print('len in : ', df.shape)

    # %%
    # # Drop dup
    # # Not include card_no since changing one card also affect another.
    # cols_drop_dup = ['aeon_id', 'input_date', 'log_type']
    # df = df.drop_duplicates(subset=cols_drop_dup, keep=False)

    # %%
    # Remove dup since making the logic to drop dup will not make sense. ~2%
    # (CHG_LIMIT contains INCREASE_LIMIT, can dedup but need to much effort)
    # Will keep max limit_amount_to since this will be used as features
    cols_drop_dup = ['aeon_id', 'card_no', 'input_date', 'log_type']
    df = df.sort_values('limit_amount_to')
    df = df.drop_duplicates(subset=cols_drop_dup, keep='last')

    # %%
    print('len drop_dup : ',df.shape)

    # %%
    dct_types = {'aeon_id':str, 'card_no':str, 'log_type':str, 'remark1':str, 
                 'limit_amount_from':float, 'limit_amount_to':float, 'input_date':str}
    df = df.astype(dct_types)

    # %%
    # Drop amount NA
    df = df[df['limit_amount_from'].notna() & df['limit_amount_to'].notna()]
    df.shape

    # %% [markdown]
    # ### Aggregate functions

    # %%
    col_idx = 'aeon_id'

    # %%
    sdf = df[df['log_type'].str.lower().str.contains('increase')]

    df_inc = sdf.groupby([col_idx]).agg(
                    count_inc = ('input_date', 'nunique'),
                    max_limit = ('limit_amount_to', 'max')).reset_index()

    df_last_inc = sdf.sort_values(['input_date','limit_amount_to']) \
                    .groupby(col_idx)[['input_date','limit_amount_to']].last() \
                    .reset_index().rename(columns={'input_date':'last_inc_limit_date',
                                                   'limit_amount_to':'last_inc_limit_amount_to'})

    # %%
    sdf = df[~df['log_type'].str.lower().str.contains('increase')]

    df_chg = sdf.groupby([col_idx]).agg(
                    count_chg = ('input_date', 'nunique')).reset_index()

    df_last_chg = sdf.sort_values('input_date') \
                    .groupby(col_idx)[['input_date']].last() \
                    .reset_index().rename(columns={'input_date':'last_chg_limit_date'})

    # %% [markdown]
    # ### 2. Card Monthly - CR

    # %%
    query=f'''
        -- Exclude : dod, not expired card
        WITH 
        card_monthly AS (
          SELECT DISTINCT period_no, aeon_id, card_no, available_cr_limit, limit_cr
          FROM card_monthly 
          WHERE period_no = '{DOD}'
        ),
        card_info AS (
          SELECT DISTINCT aeon_id, card_no14 AS card_no, expire_date
          FROM card_info
        )

        SELECT aeon_id,
            MAX(available_cr_limit) AS available_cr_limit, MAX(limit_cr) AS limit_cr
        FROM card_monthly a
        LEFT JOIN card_info b
        USING (aeon_id, card_no)
        WHERE period_no <= SUBSTRING(expire_date, 1, 6)
        GROUP BY aeon_id
    '''

    df = td.read_td_query(query, engine)

    # %%
    print('len in : ', df.shape)

    # %%
    dct_types = {'aeon_id':str, 'available_cr_limit':float, 'limit_cr':float}
    df = df.astype(dct_types)

    # %%
    df['limit_cr'] = df['limit_cr'].fillna(0)

    # If limit = 0, pct = 0 but negative keep it as -inf -> will be treated with outlier
    df['pct_available_cr_limit'] = df['available_cr_limit'] / df['limit_cr']
    df['pct_available_cr_limit'] = np.where( (df['limit_cr'] == 0) & (df['available_cr_limit'] >= 0), 0, df['pct_available_cr_limit'])
    df['pct_available_cr_limit'] = np.clip(df['pct_available_cr_limit'], -1.1, 1)

    # %%
    df_cr = df.copy()

    # %% [markdown]
    # ### 3. Card Info - YC

    # %%
    query=f'''
        SELECT aeon_id, card_no14, limit_cash, limit_shop, last_incre_limit_date, last_reduce_limit_date 
        FROM card_info
        WHERE last_incre_limit_date <= '{EOM}' AND last_reduce_limit_date <= '{EOM}'
    '''

    df = td.read_td_query(query, engine)

    # %%
    # Drop dup
    df = df.sort_values(['last_incre_limit_date'], ascending=False).drop_duplicates(['aeon_id'])

    # %%
    print('len in : ', df.shape)

    # %%
    dct_types = {'aeon_id':str, 'limit_cash':float, 'limit_shop':float,
                 'last_incre_limit_date':str, 'last_reduce_limit_date':str}
    df = df.astype(dct_types)

    # %%
    # Label date
    df['last_incre_limit_date'] = df['last_incre_limit_date'].replace('0',np.nan)
    df['last_reduce_limit_date'] = df['last_reduce_limit_date'].replace('0',np.nan)

    # df['last_incre_limit_date'] = pd.to_datetime(df['last_incre_limit_date'], format='%Y%m%d')
    # df['last_reduce_limit_date'] = pd.to_datetime(df['last_reduce_limit_date'], format='%Y%m%d')

    # df['days_from_last_sys_incre_limit'] = (eom_dt - df['last_incre_limit_date']).dt.days
    # df['days_from_last_sys_reduce_limit'] = (eom_dt - df['last_reduce_limit_date']).dt.days

    # %%
    # df = df.drop(columns=['card_no14','last_incre_limit_date','last_reduce_limit_date'])
    df = df.drop(columns=['card_no14'])

    # %%
    a = df.shape[0]
    df = df.merge(df_cr, on=col_idx, how='left') \
            .merge(df_inc, on=col_idx, how='left') \
            .merge(df_chg, on=col_idx, how='left') \
            .merge(df_last_inc, on=col_idx, how='left') \
            .merge(df_last_chg, on=col_idx, how='left')
    assert df.shape[0] == a
    del df_cr, df_inc, df_chg, df_last_inc, df_last_chg

    # %%
    df['date_of_data'] = DOD
    df.describe().T.to_csv('results/stats_account_activity_monthly.csv')
    print(df.describe().T)
    print(f'Time process : {(time.time() - st)/60} min.')

    # %%
    df.describe().T

    # %% [markdown]
    # ### Write to table

    # %%
    st = time.time()
    td.to_td(df, 'ds_feature_engineering.account_activity_monthly', con, if_exists='append', index=False)
    et = time.time()
    print(f'Time save: {(et-st)/60} min.')
    print('Run successfully')
    
if __name__=='__main__':
    MY_API_KEY = '581/8aa4544d308ca2ec6079d5dbeb8211cc1895b5f8'

    con = td.connect(apikey=MY_API_KEY, endpoint='https://api.treasuredata.co.jp')
    engine = td.create_engine("presto:dev_curated_dwh_db", con=con)

    for dod in [
        # '202401','202402','202403','202404','202405','202406','202407','202408','202409','202410',
                '202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311','202312',
               ]:
        print('='*20, dod)
        run(dod)
