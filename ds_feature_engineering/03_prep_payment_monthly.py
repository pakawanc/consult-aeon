import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import pytd.pandas_td as td

def run(DOD):
    duedate = f'{DOD}{DUEDAY}'
    currentm = datetime.strptime(DOD, "%Y%m") # datetime

    prev_dod = (currentm - relativedelta(months=1)).strftime("%Y%m") # str
    next_dod = (currentm + relativedelta(months=1)).strftime("%Y%m") # str

    st = time.time()


    # ### Payment TXN

    # In[5]:


    # payment due date at 10th
    query=f'''SELECT * FROM payment_customer
                where payment_date >= {prev_dod}{DUEDAY+1} and payment_date <= {next_dod}{DUEDAY}
            '''

    df = td.read_td_query(query, engine)


    # In[6]:


    print('len in : ', df.shape)


    # In[7]:


    # Drop dup
    cols_drop_dup = ['aeon_id', 'id_payment_seq_no', 'payment_date', 'input_time',
                       'payment_amount', 'pending_flag', 'payment_type', 'counter_no', 'input_date']
    df = df.drop_duplicates(subset=cols_drop_dup)


    # In[8]:


    print('len drop_dup : ',df.shape)


    # In[9]:


    dct_types = {'aeon_id':str, 'id_payment_seq_no':str, 'payment_date':str, 'input_time':str,
                   'payment_amount':float, 'payment_type':str }
    df = df.astype(dct_types)


    # ### Label First and Last day for DPD calculation

    # In[10]:


    df['period'] = np.where(df['payment_date'] <= duedate, 'current', 'next')


    # In[11]:


    df = df.sort_values('payment_date')
    df_last_current = df[df['period']=='current'].groupby(['aeon_id'])['payment_date'].last() \
                        .reset_index().rename(columns={'payment_date':'current_last_pmt_date'})
    df_first_next = df[df['period']=='next'].groupby(['aeon_id'])['payment_date'].first() \
                        .reset_index().rename(columns={'payment_date':'next_first_pmt_date'})


    # ### Aggregate functions

    # In[12]:


    df = df[df['period']=='current'].reset_index(drop=True)

    # Aggregate pmt amt to each day level
    df_pmt_daily = df.groupby(['aeon_id','payment_date']).agg(
                        payment_amount = ('payment_amount', 'sum')).reset_index()


    # In[14]:


    def agg_payment(df, col_idx, col_segment=None, suffix=None):
        is_segment = (col_segment is not None) and (suffix is not None)
        if is_segment:
            df = df[df[col_segment]==suffix]

        df_total = df.groupby(col_idx).agg(
                        count_pmt = ('payment_amount', 'count'),
                        sum_pmt_amt = ('payment_amount', 'sum'),
                        max_pmt_amt = ('payment_amount', 'max'),
                        min_pmt_amt = ('payment_amount', 'min'),
                        avg_pmt_amt = ('payment_amount', 'mean'),
                        md_pmt_amt = ('payment_amount', 'median'),
                    )
        if is_segment:
            df_total = df_total.rename(columns={c : f'{c}_{col_segment}_{suffix}'.lower()
                                                for c in df_total.columns})
        elif suffix is not None:
            df_total = df_total.rename(columns={c : f'{c}_{suffix}'.lower()
                                                for c in df_total.columns})

        return(df_total)

    col_idx = ['aeon_id']
    df_total = agg_payment(df, col_idx)
    df_total_daily = agg_payment(df_pmt_daily, col_idx, suffix='daily')
    df_type_4 = agg_payment(df, col_idx, col_segment='payment_type', suffix='4')


    # ### Label Payment Status

    # In[15]:


    # period_no 202410 = due date 10/10/24 -> status on this date

    # is_on_time : A,M
    # is_late : B,X
    # is_delinquent : 1-9
    # is_transactor : A,B
    # is_revolver : M,X


    # In[16]:


    query=f'''SELECT aeon_id,pay_history_all FROM payment_history_status
                where period_no = '{DOD}'
            '''

    df = td.read_td_query(query, engine)


    # In[17]:


    print('len in : ', df.shape)


    # In[18]:


    dct_types = {'aeon_id':str, 'pay_history_all':str}
    df = df.astype(dct_types)


    # In[19]:


    df['status'] = df['pay_history_all'].str[0]

    df['is_on_time'] = (df['status'].isin(['A','M'])).astype(int)
    df['is_late'] = (df['status'].isin(['B','X'])).astype(int)
    df['is_delinquent'] = (df['status'].between('1','9')).astype(int)

    df['is_transactor'] = (df['status'].isin(['A','B'])).astype(int)
    df['is_revolver'] = (df['status'].isin(['M','X'])).astype(int)

    df = df.drop(columns=['pay_history_all', 'status'])


    # In[20]:


    a = df.shape[0]
    df = df.merge(df_total, on='aeon_id', how='left') \
            .merge(df_total_daily, on='aeon_id', how='left') \
            .merge(df_type_4, on='aeon_id', how='left') \
            .merge(df_first_next, on='aeon_id', how='left') \
            .merge(df_last_current, on='aeon_id', how='left')
    assert a == df.shape[0]
    del df_total, df_total_daily, df_type_4, df_first_next, df_last_current


    # ### Label DPD

    # In[21]:


    df['next_first_pmt_date'] = pd.to_datetime(df['next_first_pmt_date'], format='%Y%m%d')
    df['current_last_pmt_date'] = pd.to_datetime(df['current_last_pmt_date'], format='%Y%m%d')

    duedate_dt = datetime.strptime(duedate, '%Y%m%d')
    # Calculate the number of days between 'date' and 'duedate'
    df['no_day_pass_due'] = np.where(df['is_late'],
                                     (df['next_first_pmt_date'] - duedate_dt).dt.days, np.nan)
    df['no_day_bf_due'] = np.where(df['is_on_time'],
                                     (df['current_last_pmt_date'] - duedate_dt).dt.days, np.nan)


    # In[22]:

    df['date_of_data'] = DOD
    df.describe().T.to_csv('results/stats_payment_history_monthly.csv')
    print(df.describe().T)
    print(f'Time process : {(time.time() - st)/60} min.')


    # ### Write to table

    # In[ ]:


    st = time.time()
    td.to_td(df, 'ds_feature_engineering.payment_history_monthly', con, if_exists='append', index=False)
    et = time.time()
    print(f'Time save: {(et-st)/60} min.')
    
    print('Run successfully')
    
if __name__=='__main__':
    MY_API_KEY = '581/8aa4544d308ca2ec6079d5dbeb8211cc1895b5f8'

    con = td.connect(apikey=MY_API_KEY, endpoint='https://api.treasuredata.co.jp')
    engine = td.create_engine("presto:dev_curated_dwh_db", con=con)

    DUEDAY = 10
    for dod in ['202301','202302','202303','202304','202305','202306','202307','202308','202309','202310','202311','202312',
                # '202401','202402','202403','202404','202405','202406','202407','202408','202409','202410','202411','202412'
               ]:
        print('='*20, dod)
        run(dod)
