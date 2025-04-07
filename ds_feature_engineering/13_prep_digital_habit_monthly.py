import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import pytd.pandas_td as td

def run(DOD):
    st = time.time()

    # %% [markdown]
    # ### 1. Online Spendings

    # %%
    query=f'''
        SELECT 
            aeon_id, card_no14, ref_no, sales_date, sale_time,
            amount, merchant_code, sub_prod_type,
            send_status, delete_flag, reverse_adj
        FROM sales_card
        WHERE SUBSTR(CAST(sales_date AS VARCHAR), 1, 6) = '{DOD}' 
            AND CAST(send_status AS INT) = 1
            AND online_flag = 'Y'
    '''

    df = td.read_td_query(query, engine)

    # %%
    print('len in : ',df.shape)

    # %%
    # Drop dup
    df = df.drop_duplicates()

    # %%
    print('len drop_dup : ',df.shape)

    # %%
    dct_types = {'aeon_id':str, 'card_no14':str, 'ref_no':str, 
                 'sales_date':int, 'sale_time':int, 'amount':float, 
                 # Keep status as str so that can handle NA
                 'send_status':str, 'delete_flag':str, 'reverse_adj':str
                }
    df = df.astype(dct_types)

    # %% [markdown]
    # ### Clean reverse txn

    # %%
    # Get txn which sum to 0 -> still has reverse txn ~10%
    col_idx = ['aeon_id','card_no14','ref_no','sales_date','sale_time']
    df_rev = df[df['reverse_adj']=='3'][col_idx]
    # print(df_rev.shape)

    df_rev = df.merge(df_rev, on=col_idx, how='inner')

    sum_rev = df_rev.groupby(col_idx)['amount'].sum()
    sum_rev = sum_rev[sum_rev == 0].reset_index().drop(columns=['amount'])
    # print(sum_rev.shape)

    # %%
    a = df.shape[0]
    df = df.merge(sum_rev, on=col_idx, how='left', indicator='check')
    assert a == df.shape[0], 'Dup rows in left table'

    # Drop rows exist in reversed df.
    df = df[df['check'] != 'both']
    df = df.drop(columns=['check'])

    # %%
    df = df[df['amount'] >= 0]

    # %%
    print('len clean : ',df.shape)

    # %% [markdown]
    # ### Aggregate functions

    # %%
    # MCC code
    # Restaurants 5811-5814,5921,5499
    # Transportation 4000-4799,3300-3499
    # Leisure 3500-3999,5733,5815-5818,5941-5946,7011,7832
    # Wellness 5122,5912,7298,7992,7997,8000-8999
    # Airline 3000-3299,4511
    # Fashion 5311,5600-5699
    df['merchant_group'] = np.where(df['merchant_code'].between('5811','5814') |
                                    df['merchant_code'].isin(['5921', '5499']),
                                    'Restaurants',
                            np.where(df['merchant_code'].between('4000','4799') |
                                     df['merchant_code'].between('3300','3499'),
                                     'Transportation',
                            np.where(df['merchant_code'].between('3500','3999') |
                                     df['merchant_code'].between('5815','5818') |
                                     df['merchant_code'].between('5941','5946') |
                                     df['merchant_code'].isin(['5733', '7011', '7832']),
                                     'Leisure',
                            np.where(df['merchant_code'].between('8000','8999') |
                                     df['merchant_code'].isin(['5122', '5912', '7298', '7992', '7997']),
                                     'Wellness',
                            np.where(df['merchant_code'].between('3000','3299') |
                                     df['merchant_code'].isin(['4511']),
                                     'Airline',
                            np.where(df['merchant_code'].between('5600','5699') |
                                     df['merchant_code'].isin(['5311']),
                                     'Fashion', np.nan))))))

    # %%
    def agg_ol_spending(df, col_idx, col_segment=None, segment=None):
        is_segment = (col_segment is not None) and (segment is not None)
        if is_segment:
            df = df[df[col_segment]==segment]

        df_total = df.groupby(col_idx).agg(
                        count_ol_spending = ('ref_no', 'count'),
                        sum_ol_amt = ('amount', 'sum'),
                        max_ol_amt = ('amount', 'max'),
                        min_ol_amt = ('amount', 'min'),
                        avg_ol_amt = ('amount', 'mean'))

        if is_segment:
            df_total = df_total.rename(columns={c:f'{c}_{col_segment}_{segment}'.lower() 
                                                for c in df_total.columns})
        return(df_total)

    col_idx = 'aeon_id'
    df_total = agg_ol_spending(df, col_idx)
    # By merchant code
    df_mc_res = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Restaurants')
    df_mc_tran = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Transportation')
    df_mc_lei = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Leisure')
    df_mc_well = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Wellness')
    df_mc_fsh = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Fashion')
    df_mc_air = agg_ol_spending(df, col_idx, col_segment='merchant_group', segment='Airline')
    # By sub_prod_type
    df_typ_jp = agg_ol_spending(df, col_idx, col_segment='sub_prod_type', segment='JP')
    df_typ_cd = agg_ol_spending(df, col_idx, col_segment='sub_prod_type', segment='CD')
    df_typ_ch = agg_ol_spending(df, col_idx, col_segment='sub_prod_type', segment='CH')

    # %%
    mcc_ls = set(df['merchant_group'].unique()) - set(['nan'])
    for mcc in mcc_ls:
        df[f'last_date_ol_merchant_group_{mcc}'.lower()] = np.where(df['merchant_group']==mcc, df['sales_date'], np.nan)

    typ_ls = ['JP','CD','CH']
    for typ in typ_ls:
        df[f'last_date_ol_sub_prod_type_{typ}'.lower()] = np.where(df['sub_prod_type']==typ, df['sales_date'], np.nan)

    # %%
    col_last_date = [c for c in df.columns if 'last_date' in c]
    df_last_date = df.sort_values(['sales_date']).groupby(col_idx)[col_last_date].last()

    # %%
    df_spending = df_total.merge(df_mc_res, on=col_idx, how='left') \
                    .merge(df_mc_tran, on=col_idx, how='left') \
                    .merge(df_mc_lei, on=col_idx, how='left') \
                    .merge(df_mc_well, on=col_idx, how='left') \
                    .merge(df_mc_fsh, on=col_idx, how='left') \
                    .merge(df_mc_air, on=col_idx, how='left') \
                    .merge(df_typ_jp, on=col_idx, how='left') \
                    .merge(df_typ_cd, on=col_idx, how='left') \
                    .merge(df_typ_ch, on=col_idx, how='left') \
                    .merge(df_last_date, on=col_idx, how='left')
    del df_total, df_mc_res, df_mc_tran, df_mc_lei, df_mc_well, df_mc_fsh, df_mc_air, df_typ_jp, df_typ_cd, df_typ_ch, df_last_date

    # %% [markdown]
    # ### 2. Promotion

    # %%
    query=f'''
        SELECT 
            aeon_id, agreement_no, keyword, sent_date, sent_time, chanel
        FROM sms2way
        WHERE SUBSTR(CAST(sent_date AS VARCHAR), 1, 6) = '{DOD}' 
    '''

    df = td.read_td_query(query, engine)

    # %%
    print('len in : ',df.shape)

    # %%
    # Drop dup
    df = df.drop_duplicates()
    print('len drop_dup : ',df.shape)

    # %%
    dct_types = {'aeon_id':str, 'chanel':str}
    df = df.astype(dct_types)

    # %%
    # Label flag
    df['is_web'] = np.where(df['chanel'].str.upper() == 'W', 1, 0)
    df['is_mobile'] = np.where(df['chanel'].str.upper() == 'M', 1, 0)
    df['is_cc'] = np.where(df['chanel'].str.upper() == 'C', 1, 0)

    # %%
    df_promo = df.groupby(col_idx).agg(
                    count_promo_web = ('is_web', 'sum'),
                    count_promo_mobile = ('is_mobile', 'sum'),
                    count_promo_call = ('is_cc', 'sum'))

    df_promo['pct_ol_promo'] = (df_promo['count_promo_web'] + df_promo['count_promo_mobile']) / df_promo.sum(axis=1)
    df_promo = df_promo.reset_index()

    # %%
    # Need outer join since they are from different base
    df = df_spending.merge(df_promo, on=col_idx, how='outer')

    # %%
    df['date_of_data'] = DOD
    print(f'len out {DOD} : {df.shape}')

    # %%
    df.describe().T.to_csv('results/stats_digital_habit_monthly.csv')
    print(df.describe().T)
    print(f'Time process : {(time.time() - st)/60} min.')

    # %%
    df.describe().T

    # %% [markdown]
    # ### Write to table

    # %%
    def out_type(col):
        if col.startswith('last_date') or col in ['aeon_id']:
            return str

        return float

    dct_dict = {c : out_type(c) for c in df.columns}
    df = df.astype(dct_dict)

    # %%
    st = time.time()
    td.to_td(df, 'ds_feature_engineering.digital_habit_monthly', con, if_exists='append', index=False)
    et = time.time()
    print(f'Time save: {(et-st)/60} min.')
    print('Run successfully')
    
if __name__=='__main__':
    MY_API_KEY = '581/8aa4544d308ca2ec6079d5dbeb8211cc1895b5f8'

    con = td.connect(apikey=MY_API_KEY, endpoint='https://api.treasuredata.co.jp')
    engine = td.create_engine("presto:dev_curated_dwh_db", con=con)

    for dod in [
        '202401','202402','202403','202404','202405','202406','202407','202408','202409',#'202410',
                '202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311','202312',
               ]:
        print('='*20, dod)
        run(dod)
