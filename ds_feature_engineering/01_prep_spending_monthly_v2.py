import os
import numpy as np
import pandas as pd
import time
import pytd.pandas_td as td

def run(DOD):
    st = time.time()

    # %%
    query=f'''
    WITH
    sales AS (
      SELECT DISTINCT
        aeon_id, card_no14, period_no, ref_no, sales_date, sale_time, amount,
        send_status, delete_flag, reverse_adj, sub_prod_type, merchant_code
      FROM sales_card
      WHERE SUBSTR(CAST(sales_date AS VARCHAR), 1, 6) = '{DOD}'
        AND send_status = 1
    )

    SELECT s.*, 
      CASE WHEN c.business_code LIKE 'Z%' THEN 'YC' ELSE 'CR' END AS card_type_f
    FROM sales s
    LEFT JOIN dev_curated_dwh_db.card_info c
    ON CAST(s.card_no14 AS VARCHAR) = c.card_no14
    -- WHERE (SUBSTRING(apply_date,1,6) <= '{DOD}' AND SUBSTRING(expire_date,1,6) >= '{DOD}')

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
    dct_types = {'aeon_id':str, 'card_no14':str, 'period_no':int, 'ref_no':str, 
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
    def agg_spending(df, col_idx, col_segment=None, segment=None, suffix=None):
        is_segment = (col_segment is not None) and (segment is not None)
        if is_segment:
            df = df[df[col_segment]==segment]

        df_total = df.groupby(col_idx).agg(
                        count_spending = ('ref_no', 'count'),
                        sum_amt = ('amount', 'sum'),
                        max_amt = ('amount', 'max'),
                        min_amt = ('amount', 'min'),
                        avg_amt = ('amount', 'mean'))

        if is_segment:
            df_total = df_total.rename(columns={c : f'{c}_{col_segment}_{segment}'.lower()
                                                for c in df_total.columns})
        if suffix is not None:
            df_total = df_total.rename(columns={c : f'{c}_{suffix}'.lower()
                                                for c in df_total.columns})

        return(df_total)

    col_idx = 'aeon_id'
    df_total = agg_spending(df, col_idx)
    # By merchant code
    df_mc_res = agg_spending(df, col_idx, col_segment='merchant_group', segment='Restaurants')
    df_mc_tran = agg_spending(df, col_idx, col_segment='merchant_group', segment='Transportation')
    df_mc_lei = agg_spending(df, col_idx, col_segment='merchant_group', segment='Leisure')
    df_mc_well = agg_spending(df, col_idx, col_segment='merchant_group', segment='Wellness')
    df_mc_fsh = agg_spending(df, col_idx, col_segment='merchant_group', segment='Fashion')
    df_mc_air = agg_spending(df, col_idx, col_segment='merchant_group', segment='Airline')
    # By sub_prod_type
    df_typ_jp = agg_spending(df, col_idx, col_segment='sub_prod_type', segment='JP')
    df_typ_cd = agg_spending(df, col_idx, col_segment='sub_prod_type', segment='CD')
    df_typ_ch = agg_spending(df, col_idx, col_segment='sub_prod_type', segment='CH')

    # %%
    mcc_ls = set(df['merchant_group'].unique()) - set(['nan'])
    for mcc in mcc_ls:
        df[f'last_date_merchant_group_{mcc}'.lower()] = np.where(df['merchant_group']==mcc, df['sales_date'], np.nan)

    typ_ls = ['JP','CD','CH']
    for typ in typ_ls:
        df[f'last_date_sub_prod_type_{typ}'.lower()] = np.where(df['sub_prod_type']==typ, df['sales_date'], np.nan)

    col_last_date = [c for c in df.columns if 'last_date' in c]
    df_last_date = df.sort_values(['sales_date']).groupby(col_idx)[col_last_date].last()

    # %%
    df_res = df_total.merge(df_mc_res, on=col_idx, how='left') \
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
    # ### CR Card

    # %%
    card_type_f = 'CR'
    dfcr = df[df['card_type_f'] == card_type_f].copy()

    col_idx = 'aeon_id'
    df_total = agg_spending(dfcr, col_idx, suffix=card_type_f)
    # By merchant code
    df_mc_res = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Restaurants', suffix=card_type_f)
    df_mc_tran = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Transportation', suffix=card_type_f)
    df_mc_lei = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Leisure', suffix=card_type_f)
    df_mc_well = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Wellness', suffix=card_type_f)
    df_mc_fsh = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Fashion', suffix=card_type_f)
    df_mc_air = agg_spending(dfcr, col_idx, col_segment='merchant_group', segment=f'Airline', suffix=card_type_f)
    # By sub_prod_type
    df_typ_jp = agg_spending(dfcr, col_idx, col_segment='sub_prod_type', segment=f'JP', suffix=card_type_f)
    df_typ_cd = agg_spending(dfcr, col_idx, col_segment='sub_prod_type', segment=f'CD', suffix=card_type_f)
    df_typ_ch = agg_spending(dfcr, col_idx, col_segment='sub_prod_type', segment=f'CH', suffix=card_type_f)

    # %%
    mcc_ls = set(dfcr['merchant_group'].unique()) - set(['nan'])
    for mcc in mcc_ls:
        dfcr[f'last_date_merchant_group_{mcc}_{card_type_f}'.lower()] = np.where(dfcr['merchant_group']==mcc, dfcr['sales_date'], np.nan)

    typ_ls = ['JP','CD','CH']
    for typ in typ_ls:
        dfcr[f'last_date_sub_prod_type_{typ}_{card_type_f}'.lower()] = np.where(dfcr['sub_prod_type']==typ, dfcr['sales_date'], np.nan)

    col_last_date = [c for c in dfcr.columns if 'last_date' in c and card_type_f.lower() in c]
    df_last_date = dfcr.sort_values(['sales_date']).groupby(col_idx)[col_last_date].last()

    # %%
    df_res_cr = df_total.merge(df_mc_res, on=col_idx, how='left') \
                    .merge(df_mc_tran, on=col_idx, how='left') \
                    .merge(df_mc_lei, on=col_idx, how='left') \
                    .merge(df_mc_well, on=col_idx, how='left') \
                    .merge(df_mc_fsh, on=col_idx, how='left') \
                    .merge(df_mc_air, on=col_idx, how='left') \
                    .merge(df_typ_jp, on=col_idx, how='left') \
                    .merge(df_typ_cd, on=col_idx, how='left') \
                    .merge(df_typ_ch, on=col_idx, how='left') \
                    .merge(df_last_date, on=col_idx, how='left')
    del dfcr, df_total, df_mc_res, df_mc_tran, df_mc_lei, df_mc_well, df_mc_fsh, df_mc_air, df_typ_jp, df_typ_cd, df_typ_ch, df_last_date

    # %% [markdown]
    # ### YC Card

    # %%
    card_type_f = 'YC'
    dfyc = df[df['card_type_f'] == card_type_f].copy()

    col_idx = 'aeon_id'
    df_total = agg_spending(dfyc, col_idx, suffix=card_type_f)
    # By merchant code
    df_mc_res = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Restaurants', suffix=card_type_f)
    df_mc_tran = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Transportation', suffix=card_type_f)
    df_mc_lei = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Leisure', suffix=card_type_f)
    df_mc_well = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Wellness', suffix=card_type_f)
    df_mc_fsh = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Fashion', suffix=card_type_f)
    df_mc_air = agg_spending(dfyc, col_idx, col_segment='merchant_group', segment=f'Airline', suffix=card_type_f)
    # By sub_prod_type
    df_typ_jp = agg_spending(dfyc, col_idx, col_segment='sub_prod_type', segment=f'JP', suffix=card_type_f)
    df_typ_cd = agg_spending(dfyc, col_idx, col_segment='sub_prod_type', segment=f'CD', suffix=card_type_f)
    df_typ_ch = agg_spending(dfyc, col_idx, col_segment='sub_prod_type', segment=f'CH', suffix=card_type_f)

    # %%
    mcc_ls = set(dfyc['merchant_group'].unique()) - set(['nan'])
    for mcc in mcc_ls:
        dfyc[f'last_date_merchant_group_{mcc}_{card_type_f}'.lower()] = np.where(dfyc['merchant_group']==mcc, dfyc['sales_date'], np.nan)

    typ_ls = ['JP','CD','CH']
    for typ in typ_ls:
        dfyc[f'last_date_sub_prod_type_{typ}_{card_type_f}'.lower()] = np.where(dfyc['sub_prod_type']==typ, dfyc['sales_date'], np.nan)

    col_last_date = [c for c in dfyc.columns if 'last_date' in c and card_type_f.lower() in c]
    df_last_date = dfyc.sort_values(['sales_date']).groupby(col_idx)[col_last_date].last()

    # %%
    df_res_yc = df_total.merge(df_mc_res, on=col_idx, how='left') \
                    .merge(df_mc_tran, on=col_idx, how='left') \
                    .merge(df_mc_lei, on=col_idx, how='left') \
                    .merge(df_mc_well, on=col_idx, how='left') \
                    .merge(df_mc_fsh, on=col_idx, how='left') \
                    .merge(df_mc_air, on=col_idx, how='left') \
                    .merge(df_typ_jp, on=col_idx, how='left') \
                    .merge(df_typ_cd, on=col_idx, how='left') \
                    .merge(df_typ_ch, on=col_idx, how='left') \
                    .merge(df_last_date, on=col_idx, how='left')
    del dfyc, df_total, df_mc_res, df_mc_tran, df_mc_lei, df_mc_well, df_mc_fsh, df_mc_air, df_typ_jp, df_typ_cd, df_typ_ch, df_last_date

    # %%
    a = df_res.shape[0]
    df_res = df_res.merge(df_res_cr, on=col_idx, how='left') \
                    .merge(df_res_yc, on=col_idx, how='left')
    assert df_res.shape[0] == a

    # %%
    df_res['date_of_data'] = DOD

    col_date = [c for c in df_res.columns if 'last_date' in c]
    for c in col_date:
        df_res[c] = pd.to_datetime(df_res[c], format='%Y%m%d').dt.strftime('%Y%m%d')

    print(f'len out {DOD} : {df_res.shape}')

    # %%
    df_res.describe().T.to_csv('results/stats_spending_habits_monthly_v2.csv')
    print(df_res.describe().T)
    print(f'Time process : {(time.time() - st)/60} min.')

    # %% [markdown]
    # ### Write to table

    # %%
    st = time.time()
    td.to_td(df_res, 'ds_feature_engineering.spending_habits_monthly_v2', con, if_exists='append', index=False)
    et = time.time()
    print(f'Time save: {(et-st)/60} min.')
    print('Run successfully')
    
if __name__=='__main__':
    MY_API_KEY = '581/8aa4544d308ca2ec6079d5dbeb8211cc1895b5f8'

    con = td.connect(apikey=MY_API_KEY, endpoint='https://api.treasuredata.co.jp')
    engine = td.create_engine("presto:dev_curated_dwh_db", con=con)

    for dod in ['202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311','202312',
                '202401','202402','202403','202404', '202405','202406','202407','202408','202409',#'202410','202411','202412'
               ]:
        print('='*20, dod)
        run(dod)
