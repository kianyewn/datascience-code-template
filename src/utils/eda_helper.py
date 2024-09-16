import pickle
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
import os
import numpy as np
import datetime
import pandas as pd
import yaml
from typing import Union, List, Dict
import tqdm
from collections import defaultdict
import decimal

F = lambda x: x

class yaml_helper:
    @staticmethod
    def load_yaml(yaml_path:str):
        with open(yaml_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        print(f'{yaml_path} loaded.')
        return config
   
    @staticmethod
    def write_to_yaml(obj: Dict, yaml_path:str):
        with open(yaml_path, 'w') as stream:
            yaml.dump(obj, stream)
        print('Saved to  {yaml_path]}')
        return
    
    
def agg_numerical_features(df, key, numerical_features):
    # df is pyspark df
    funcs = {'min':F.min, 'mean': F.mean, 'max':F.max, 'std':F.stddev, 'sum': F.sum}
    agg_funcs = []
    for col in numerical_features:
        agg_func = [function(col).alias(f'{col}_{func_name}') if func_name != 'median'
                   else function.alias(f'{col}_{func_name}') for func_name, function in funcs.items()]
        agg_funcs.extend(agg_func)
    return df.groupby(key).agg(*agg_funcs)

def toPandas(spark_df, pkey=None):
    from collections import Counter
    c = Counter(spark_df.columns)
    dups = [n for n,f in c.items() if f > 1]
    print('Duplicated column names: {dups}')
    for col, dtype in spark_df.dtypes:
        if 'date' in dtype or 'time' in dtype:
            spark_df = spark_df.withColumn(col, F.col(col).cast('string'))
    pd_df = spark_df.toPandas()
    if pkey:
        pd_df['distinct_cnt_{pkey}'] = pd_df[pkey].nunique(dropna=False) # by default dropna is set to True
        pd_df['records'] = pd_df.shape[0]
    return pd_df

def get_cumsum(frequency_table, count_col):
    frequency_table['perc'] = frequency_table[count_col] / frequency_table[count_col].sum()
    frequency_table['cum_perc'] = frequency_table['perc'].cumsum()
    frequency_table['cum_perc'].plot()
    return frequency_table

def rename_columns(include_cols, prefix ='sa_dly_'):
    renamed_columns = ['party_id','as_of_dt'] + ['sa_dly_' + col for col in include_cols if not (col=='party_id' or col == 'as_of_dt')]
    return renamed_columns

def convert_day_to_week(pd_sample, date_col='date'):
    min_date = pd_sample[date_col].min()
    max_date = pd_sample[date_col].max()
    print(f'min_date: {min_date}, max_date: {max_date}')
    date_range = pd.date_range(start=min_date, end=max_date, freq='7d')
    weeks = {}
    for idx, (start_date, end_date) in enumerate(zip(date_range[:-1], date_range[1:])):
        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
        weeks[f'{start_date}_{end_date}'] = f'week_{idx+1:02}_[start]_{end}'
    
    def map_week(x):
        for key, value in weeks.items():
            week_start_date, week_end_date = key.split('_')
            if (x>=week_start_date) and (x < week_end_date):
                return value
            return 'missing'
    
    pd_sample['week'] = pd_sample['date_col'].apply(lambda x: map_week(x))
    pd_sample2 = pd_sample[pd_sample['week'].apply(lambda x: x!='missing')]
    return pd_sample2

def convert_day_to_month(pd_sample, date_col='date'):
    min_date = pd_sample[date_col].min()
    max_date = pd_sample[date_col].max()
    
    date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
    
    months = {}
    for idx, (start_date, end_date) in enumerate(zip(date_range[:-1], date_range[1:])):
        start = start_date.date().strftime('%Y-%m-%d')
        end = end_date.date().strftime('%Y-%m-%d')
        months[f'{start}_{end}'] = f'months_{idx+1:02}_{start}_{end}'
        
    def map_month(x):
        for key, value in months.items():
            month_start_date, month_end_date = key.split('_')
            if (x>=month_start_date) and (x < month_end_date):
                return value
        return 'missing'
    pd_sample['month'] = pd_sample[date_col].apply(lambda x: map_month(x))
    pd_sample2 = pd_sample[pd_sample['month'].apply(lambda x: x != 'missing')]
    return pd_sample2

def get_psi(stat_ratio):
    eps = np.finfo(np.float32).eps
    psi_list = []
    label_psi = []
    
    # calc psi
    for idx, col in enumerate(stat_ratio.columns):
        if idx == 0 or idx==len(stat_ratio.columns)-1:
            continue
        prev = stat_ratio.iloc[:, idx-1]
        curr = stat_ratio.iloc[:, idx]
        psi = (curr - prev) * np.log((curr+eps) / (prev+eps))
        psi_sum = psi.sum()
        psi_list.append(psi.apply(lambda x: round(x,4)))
        label_psi.append(psi_sum)
    
    # add entire column psi
    label_psi_s = pd.Series([np.nan] + label_psi + [np.nan], index=stat_ratio.columns)
    # dummy fill first and last column
    nans = pd.Series((stat_ratio.shape[0]) * [np.nan], index=stat_ratio.index.tolist())
    psi_list = [nans] + psi_list + [nans]
    
    # get dataframe of psi
    psi_df = pd.concat(psi_list, axis=1)
    psi_df.columns = stat_ratio.columns
    # rename index
    psi_df = psi_df.drop('All')
    psi_df.loc['all_unique_labels',:] = label_psi_s
    
    og_index_order = psi_df.index.tolist()
    reindex_order = og_index_order[-1:] +  og_index_order[:-1]
    psi_df = psi_df.reindex(reindex_order)
    psi.columns.name = None
    return psi_df.reset_index()

class ExcelWriter:
    def __init__(self, base_folder, filename, overwrite=False):
        self.base_folder = base_folder
        self.filename = filename
        
        # create nested Excel file if it does not exist
        os.makedirs(base_folder, exist_ok=True)
        
        # create excel file if it does not exist
        if not os.path.exiss(f"{base_folder}/{filename}.xlsx") or overwrite:
            pd.DataFrame().to_excel(f'{base_folder}/{filename}.xlsx', index=False)
            
    def write_to_excel(self, sheet_name, df):
        with pd.ExcelWriter(f'{self.base_folder}/{self.filename}.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            # write the dataframe to specific sheet
            df.to_excel(excel_writer=writer, sheet_name=sheet_name, index=False)
            

def gen_label_report(df_pd, writer, cust_key, label, date_col):
    """
    from src.utils.excel_writer import ExcelWriter
    writer = ExcelWriter(base_folder='data/samples/', filename='mvp2_intents_labels_summary')
    labels_pd is the full data (not sampled)
    gen_label_report(labels_pd, writer, 'cif_id', 'label', 'date')

    Args:
        df_pd (_type_): _description_
        writer (_type_): _description_
        cust_key (_type_): _description_
        label (_type_): _description_
        date_col (_type_): _description_
    """
    stat_total_dly = pd.pivot_table(df_pd, values=cust_key, index=[label], columns=[date_col], aggfunc='count', fill_value=0, margins=True)
    stat_total_week = pd.pivot_table(df_pd, values=cust_key, index=[label], columns=['week'], aggfunc='count', fill_value=0, margins=True)
    stat_total_month = pd.pivot_table(df_pd, values=cust_key, index=[label], columns=['month'], aggfunc='count', fill_value=0, margins=True)
    
    writer.write_to_excel(sheet_name='total_dly', df=stat_total_dly.reset_index())
    writer.write_to_excel(sheet_name='total_wly', df=stat_total_week.reset_index())
    writer.write_to_excel(sheet_name='total_mly', df=stat_total_month.reset_index())
    
    stat_ratio_dly = stat_total_dly.div(stat_total_dly.iloc[-1])
    stat_ratio_week = stat_total_week.div(stat_total_week.iloc[-1])
    stat_ratio_month = stat_total_dly.div(stat_total_month.iloc[-1])
    
    writer.write_to_excel(sheet_name='ratio_dly', df=stat_ratio_dly.reset_index())
    writer.write_to_excel(sheet_name='ratio_wly', df=stat_ratio_week.reset_index())
    writer.write_to_excel(sheet_name='ratio_mly', df=stat_ratio_month.reset_index())

    psi_dly = get_psi(stat_ratio_dly)
    psi_wly = get_psi(stat_ratio_week)
    psi_mly = get_psi(stat_ratio_month)
 
    writer.write_to_excel(sheet_name='psi_dly', df=psi_dly.reset_index())
    writer.write_to_excel(sheet_name='psi_wly', df=psi_wly.reset_index())
    writer.write_to_excel(sheet_name='psi_mly', df=psi_mly.reset_index())
    return


def get_data_summary(df):
    cols = []
    num_uniques = []
    samples = []
    types = []
    proposed_data_types = []
    non_missing_cnts = []
    missing_cnts = []
    missing_rates = []
    zero_cnts = []
    zero_rates = []
    num_rows = df.shape[0]
    for col in tqdm.tqdm(df.columns):
        cols.append(col)
        
        # num unique
        num_unique = df[col].nunique(dropna=False)
        num_uniques.append(num_unique)
        
        # samples
        unique = df[col].unique().tolist()
        candidates = list(set(unique[:5] + unique[-5:]))
        samples.apppend(candidates)
        dtype = set(map(lambda x: type(x), candidates))
        types.append(dtype)
        
        # proposed data type
        candidates_str = [str(x) for x in candidates]
        proposed_data_type = 'numerical' if decimal.Decimal in dtype or int in dtype or (float in dtype and str not in dtype) else 'categorical'
        proposed_data_types.append(proposed_data_type)
        
        # non missing and missing cnts
        missing_cnt = df[col].apply(lambda x: pd.isna(x)).sum()
        missing_cnts.append(missing_cnt)
        missing_rate = missing_cnt / num_rows
        non_missing_cnts.append(num_rows-missing_cnt)
        missing_rates.append(missing_rate)
        
        # zero counts
        if proposed_data_type == 'numerical':
            zero_cnt = df[col].apply(lambda x: x==0).sum()
            zero_rate = zero_cnt / num_rows
            zero_cnts.append(zero_cnt)
            zero_rate.append(zero_rate)
        else:
            # if categorical, zero cnt and rates are not applicable
            zero_cnts.append(np.nan)
            zero_rates.append(np.nan)
    summary = pd.DataFrame({'var': cols,
                            'num_unique': num_uniques,
                            'samples': samples,
                            'type': types,
                            'proposed_data_type': proposed_data_types,
                            'non_missing_cnt': non_missing_cnts,
                            'missing_cnt': missing_cnts,
                            'missing_rate': missing_rates,
                            'zero_cnt': zero_cnts,
                            'zero_rate': zero_rates,
                            })
    # Add summary
    summary['remarks'] = ''
    summary['remarks'] = summary.apply(lambda x: f"{x['remarks']}, missing_rate==1" if x['missing_rate']==1 else x['remarks'], axis=1)
    summary['remarks'] = summary.apply(lambda x: f"{x['remarks']}, zero_rate==1" if x['zero_rate']==1 else x['remarks'], axis=1)
    summary['remarks'] = summary.aply(lambda x: f"{x['remarks']}, cardinality==1" if x['num_unique']==1 else x['remarks'], axis=1)
    return summary   
    
    

def get_data_summary(df):
    cols = []
    num_uniques = []
    samples = []
    types = []
    proposed_data_types = []
    non_missing_cnts = []
    missing_cnts = []
    missing_rates = []
    zero_cnts = []
    zero_rates = []
    num_rows = df.shape[0]
    for col in tqdm.tqdm(df.columns):
        cols.append(col)
        
        # num unique
        num_unique = df[col].nunique(dropna=False)
        num_uniques.append(num_unique)
        
        # samples
        unique = df[col].unique().tolist()
        candidates = list(set(unique[:5] + unique[-5:]))
        samples.apppend(candidates)
        dtype = set(map(lambda x: type(x), candidates))
        types.append(dtype)
        
        # proposed data type
        candidates_str = [str(x) for x in candidates]
        proposed_data_type = 'numerical' if decimal.Decimal in dtype or int in dtype or (float in dtype and str not in dtype) else 'categorical'
        proposed_data_types.append(proposed_data_type)
        
        # non missing and missing cnts
        missing_cnt = df[col].apply(lambda x: pd.isna(x)).sum()
        missing_cnts.append(missing_cnt)
        missing_rate = missing_cnt / num_rows
        non_missing_cnts.append(num_rows-missing_cnt)
        missing_rates.append(missing_rate)
        
        # zero counts
        if proposed_data_type == 'numerical':
            zero_cnt = df[col].apply(lambda x: x==0).sum()
            zero_rate = zero_cnt / num_rows
            zero_cnts.append(zero_cnt)
            zero_rate.append(zero_rate)
        else:
            # if categorical, zero cnt and rates are not applicable
            zero_cnts.append(np.nan)
            zero_rates.append(np.nan)
    summary = pd.DataFrame({'var': cols,
                            'num_unique': num_uniques,
                            'samples': samples,
                            'type': types,
                            'proposed_data_type': proposed_data_types,
                            'non_missing_cnt': non_missing_cnts,
                            'missing_cnt': missing_cnts,
                            'missing_rate': missing_rates,
                            'zero_cnt': zero_cnts,
                            'zero_rate': zero_rates,
                            })
    # Add summary
    summary['remarks'] = ''
    summary['remarks'] = summary.apply(lambda x: f"{x['remarks']}, missing_rate==1" if x['missing_rate']==1 else x['remarks'], axis=1)
    summary['remarks'] = summary.apply(lambda x: f"{x['remarks']}, zero_rate==1" if x['zero_rate']==1 else x['remarks'], axis=1)
    summary['remarks'] = summary.aply(lambda x: f"{x['remarks']}, cardinality==1" if x['num_unique']==1 else x['remarks'], axis=1)
    return summary

def characteristics_of_iv():
    desc = ("""
            IV rule of thumb:
                - `< 0.02`: uselesss for prediction
                - `0.02~0.1`: weak predictor
                -  `0.1 ~ 0.3`: medium predictor
                - `0.3 ~ 0.5` : strong predictor
                - `> 0.5`: too good to be true predictors
            
            Characteristics of IV
            - Usually IV will be very high for bins that have 0 pos_rate. That is why for binning, we need to ensure that bins
            contain some positive labels
            -- But need to make sure bins contain at least 5% of the dataset
            -- But iwth pandas qcut, it is also possible to have bins less than 5% of the dataset due to skewness in the data
            
            - IV will be very close to 0 because of very high zero rate
            - remove data that have only one null value
            - Suppose the sampled_data is representative of the underlying distribution
                - we can be more conservative on the feature importance of the sample, i.e. useless for prediction < 0.05
            - For categorical columns with very high cardinality, the IVs tend to be very high because IV is the sum of individual cardinalities
            - It is best to remove this categorical variables, or do some processing to remove this categories
            
            Some things to not when doing the implementation:
                - when using value_counts, dropna=True is set as default, resulting in total_pos and total_neg to be different across different features
                - when using nunique(), dropna=True is set as default, resulting in 1 smaller unique values, which can be detrimental if we are removing columns
                  with only one unique value. (i.e. it should be two values, null and non null.)
            """)
    return

def get_stat(pd_df, col, label_col, label, data_summary):
    """pd_df is binned dataframe"""
    pd_df = pd_df.copy()
    pd_df['pseudo_label'] = pd_df[label_col].apply(lambda x: 1 if x == label else 0)
    stat = pd_df.groupby([col])['pseudo_label'].value_counts(dropna=False).unstack().reset_index().rename(columns={0:'neg', 1:'pos', col:'bin'}).fillna(0)
    stat['bin'] = stat['bin'].apply(str)
    
    if '(' in stat['bin'].iloc[0] and ']' in stat['bin'].iloc[0]:
        stat['bin2'] = stat['bin'].apply(lambda x: float(x.split(',')[0][1:]) if x!='nan' else 999999999)
        stat = stat.sort_values(by=['bin2']).drop('bin2', axis=1)
    else:
        stat = stat.sort_values(by=['bin'])
    if 'pos' not in stat.columns:
        stat['pos'] = 0
    if 'neg' not in stat.columns:
        stat['neg'] = 0
    stat.columns.name = None
    
    stat['total_neg'] = stat['neg'].sum()
    stat['total_pos'] = stat['pos'].sum()
    stat['neg_rate'] = stat['neg'] / stat['total_neg']
    stat['pos_rate'] = stat['pos'] / stat['total_pos']
    stat['var'] = col
    eps = np.finfo(float).eps
    stat[f'iv'] = (stat['neg_rate']- stat['pos_rate']) * np.log((stat['neg_rate'] + eps) / (stat['pos_rate'] + eps))
    iv = stat['iv'],sum()
    stat['proposed_data_type'] = data_summary.loc[data_summary['var']==col, 'proposed_data_type'].iloc[0]
    stat['label'] = label
    
    columns_to_use = ['label','var','proposed_data_type', 'total_neg', 'total_pos']
    stat_single = stat.loc[0:0, columns_to_use]
    stat_single['iv'] = iv
    return stat_single

def get_stat_verbose(pd_df, col, label_col, label, data_summary):
    """pd_df is binned dataframe"""
    pd_df = pd_df.copy()
    pd_df['pseudo_label'] = pd_df[label_col].apply(lambda x: 1 if x == label else 0)
    stat = pd_df.groupby([col])['pseudo_label'].value_counts(dropna=False).unstack().reset_index().rename(columns={0:'neg', 1:'pos', col:'bin'}).fillna(0)
    stat['bin'] = stat['bin'].apply(str)
    
    if '(' in stat['bin'].iloc[0] and ']' in stat['bin'].iloc[0]:
        stat['bin2'] = stat['bin'].apply(lambda x: float(x.split(',')[0][1:]) if x!='nan' else 999999999)
        stat = stat.sort_values(by=['bin2']).drop('bin2', axis=1)
    else:
        stat = stat.sort_values(by=['bin'])
    if 'pos' not in stat.columns:
        stat['pos'] = 0
    if 'neg' not in stat.columns:
        stat['neg'] = 0
    stat.columns.name = None
    
    stat['total_neg'] = stat['neg'].sum()
    stat['total_pos'] = stat['pos'].sum()
    stat['neg_rate'] = stat['neg'] / stat['total_neg']
    stat['pos_rate'] = stat['pos'] / stat['total_pos']
    stat['var'] = col
    eps = np.finfo(float).eps
    stat[f'iv'] = (stat['neg_rate']- stat['pos_rate']) * np.log((stat['neg_rate'] + eps) / (stat['pos_rate'] + eps))
    stat['proposed_data_type'] = data_summary.loc[data_summary['var']==col, 'proposed_data_type'].iloc[0]
    stat['label'] = label

    columns_order = ['label', 'var','bin','proposed_data_type'] + [col for col in stat.columns if col not in ['label', 'var','bin', 'proposed_data_type']]
    stat = stat.loc[:, columns_order]
    return stat
    
def get_numerical_features(df):
    numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
    return numerical_features

def get_categorical_features(df):
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_features

def get_stat_main(df, data_summary):
    numerical_features = get_numerical_features(df) # remember to exclude primary key columns
    categorical_features = get_categorical_features(df)
    stat_list = []
    stat_list_verbose = []
    for lab in tqdm.tqdm(df['next_7_day_labels'].unique()):
        df_copy = df.copy()
        for col in numerical_features:
            if decimal.Decimal in data_summary.loc[data_summary['var']==col]['type'].iloc[0]:
                df_copy[col] = df_copy[col].apply(lambda x: float(x) if (str(x)!='np.nan' and x != None) else x)
            if data_summary.loc[data_summary['var']==col, 'num_unique'].iloc[0] > 10:
                df_copy[col] = pd.qcut(df_copy[col], q=10, duplicates='drop').astype(str)
            else:
                df_copy[col] = df_copy[col].fillna(9999999) # fill missing with 9999999
            num_stat_verbose = get_stat_verbose(df_copy, col, 'next_7_day_labels', lab, data_summary=data_summary)
            stat_list_verbose.append(num_stat_verbose)
            
            num_stat = get_stat(df_copy, col, 'next_7_day_labels', lab, data_summary=data_summary)
            stat_list.append(num_stat)
        for col in categorical_features:
            df_copy[col] = df_copy[col].apply(str)
            df_copy[col] = df_copy[col].apply(lambda x: x if str(x) != 'np.nan' and str(x) !=None else 'NA')
            cat_stat_verbose = get_stat_verbose(df_copy, col, 'next_7_day_labels', lab, data_summary=data_summary)
            stat_list_verbose.append(cat_stat_verbose)
            cat_stat = get_stat(df_copy, col, label_col='next_7_day_labels', label=lab, data_summary=data_summary)
            stat_list.append(cat_stat)
    # (all_stat_verbose, all_stat)
    return pd.concat(stat_list_verbose), pd.concat(stat_list)
            
def stat_feature_analysis(sample_pd, all_stat, all_stat_verbose):
    """Sample of some htings to check when doing feature analysis by IV
    """
    all_stat = all_stat_verbose # use all stat verbose to look at the bins
    label_vc = sample_pd['next_7_day_labels'].value_counts()
    
    # ignore importance of features on non-significant labels
    non_sigfig_labels = label_vc[label_vc < 100].index.tolist() # labels that are non-significant
    all_stat_copy = all_stat.copy()
    all_stat_copy.loc[all_stat_copy['label'].apply(lambda x: x in non_sigfig_labels), 'iv'] = 0
    
    agg_stat = all_stat.groupby('var')['iv'].describe() # count, mean, std, min, 25% .... max
    agg_stat.columns = [f'iv_{col}' for col in agg_stat]
    agg_stat = agg_stat.reset_index().rename(columns={'iv_count': 'num_unique_labels'})
    
    # Check 1: Can look at feature with very high iv
    ## Can see that iv is very high for labels that have little to no samples (pos is small)
    all_stat[all_stat['iv']>24]
    all_stat[all_stat['iv']==all_stat['iv'].max()] # bins that have highest iv
    
    # Check 2: remove non sigfig labels
    agg_stat2 = all_stat_copy[~all_stat_copy['label'].apply(lambda x: x in non_sigfig_labels)].groupby('var')['iv'].describe()
    agg_stat2.columns = [f'iv_{col}' for col in agg_stat2]
    agg_stat2 = agg_stat2.reset_index().rename(columns={'iv_count': 'num_unique_labels'})
    agg_stat2[agg_stat2['iv_max']>0.02] # iv more than this value
    agg_stat2[agg_stat2['iv_max']]
    return
    

class SampledDataset:
    """Generate sampled datasets for analysis"""
    def __init__(self):
        pass
    def generate_samples_labels(self, modelling_data):
        start_date, end_date = '2022-07-01', '2023-07-10'
        modelling_data = modelling_data.filter(F.col('date_col').between(start_date, end_date)) # dataset containing positive interactions (cin, date, label)
        # Hive timestamp is interpreted as UNIX timestamp in seconds
        days = lambda i: i * 86400
        Window = lambda x: x
        look_ahead_window = Window.partitionBy('user_id').orderBy(F.col('unix_date')).rangeBetween(0, days(1+7))
        look_ahead_gap_window = Window.partitionBy('user_id').orderBy(F.col('unix_date').rangeBetween(days(3), days(3+7)))
        
        modelling_data  = modelling_data.withColumn('next_7_day_labels', F.collect_set('intent').over(look_ahead_window))
        modelling_data = modelling_data.withColumn('next_7_day_labels_with_gap', F.collect_set('intent').over(look_ahead_gap_window))
        label_data = modelling_data.select('user_id','date','next_7_day_labels', F.explode('next_7_day_labels').alias('label')).drop_duplicates()
        sampling = True
        spark = lambda x: x
        if sampling == True:
            mapping_table_with_demo_path = f's3a://'
            cifs = spark.read.parquet(mapping_table_with_demo_path)
            label_data_with_pkeys = label_data.join(cifs, label_data['user_id']==cifs['kais'], 'inner').persist()
            # it is possible for some customers to not be in the master table.
            label_data_with_pkeys_final = label_data_with_pkeys.filter(F.col('cif_id').isNotNull())
            
            # Added to use pandas to sample according to sample size
            # Should sample CINs that interacted with the labels
            # Impossible to include all the features due to the data size. Therefore recommended to do feature analysis on a subset of the data.
            # distribution can be different from the training data. At this stage, it is just to access the feature importance
            # after doing the initial feature selection, will need to do further feature selection anyway
            SAMPLE_SIZE = 200
            for col, dtype in label_data_with_pkeys_final.dtypes:
                if 'date' in dtype or 'time' in dtype:
                    label_data_with_pkeys_final = label_data_with_pkeys_final.withColumn(col, F.col(col).cast('string'))
            label_data_pd = label_data_with_pkeys_final.toPandas()
            sample_pds = []
            for lab in label_data_pd['label'].unique():
                sample_base = label_data_pd[label_data_pd['label']==lab]
                sample_size = min(SAMPLE_SIZE, sample_base.shape[0])
                sample = sample_base.sample(sample_size)
                sample_pds.append(sample)
                
            sample_pd = pd.concat(sample_pds)
            label_data_with_pkeys_final = label_data.filter(F.col('cif_id').isNotNull())
        return label_data_with_pkeys_final            
    
    def load_demo_sample(self, demo):
        start_date, end_date = '2022-07-01', '2023-07-10'
        pkeys = ['cif_id']
        date_col = ['as_of_dt']
        columns = ['age','bal']
        oc_cols = pkeys + date_col + columns
        demo = demo.filter(F.col('as_of_dt').between(start_date, end_date)).select(*oc_cols)
        
        # Load this from pyspark parquet
        labels_sample = self.generate_sample_labels(modelling_data=None)
        demo_ft = labels_sample.join(demo, ['cif_id', 'date'], 'left')
        
        # forward fill
        Window = lambda x: x
        lookback_window = Window.partitionBy('cif_id').orderBy('date').rowsBetween(Window.unboundedPreceding, 0)
        for col in demo.columns:
            if col not in ['cif_id','date']:
                demo_ft = demo_ft.withColumn(col, F.last(F.col(col), ignore_nulls=True).over(lookback_window))
        return
    
    def load_historical_interactions(self, cb_int):
        columns_to_use = ['user_id','intent','session_id', 'request_timestamp', 'channel_name', 'businessdate']
        start_date, end_date = '2022-07-01', '2023-07-10'
        start_date = pd.to_datetime(start_date) + datetime.timedelta(days=-60)
        raw_chatbot = cb_int.filter(F.col('businessdate')>=start_date).filter(F.col('businessdate')<end_date).select(*columns_to_use)
        raw_chatbot = raw_chatbot.withColumn('date', F.to_date(F.to_timestamp(F.col('request_timestamp'))))
        raw_chatbot = raw_chatbot.withColumn('unix_date', F.unix_timestamp(F.col('date')))
        
        # Hive timestamp is interpreted as unix timestamp in seconds
        days = lambda i: i * 86400
        Window = lambda x: x
        w = Window.partitionBy('user_id').orderBy('unix_date').rangeBetween(-days(60), -days(1))
        raw_chatbot = raw_chatbot.withColumn('historical_interactions', F.collect_list(F.col('intent').over(w)))
        
        chatbot_columns = raw_chatbot.columns
        raw_chatbot = raw_chatbot.select(*chatbot_columns, F.explode_outer('historical_interactions').alias('cb_past_intent'))
        
        # Ensure the wide transformation contains all the values listed in pivot_list
        pivot_list = ['val1','val2'] + ['null'] 
        chatbot = raw_chatbot.groupy('user_id','date').pivot('past_cb_intent', pivot_list).agg(F.count(F.col('user_id'))).fillna(0)
        
        labels = self.generate_samples_labels(modelling_data=None) # load from parquet
        raw_chatbot = labels.join(chatbot, ['user_id'], ['date']) # inner join, because we just store the dates that have chatbot features
        # After we join we know if there are nulls or non-nulls
        return raw_chatbot

    def load_non_financial_transactions(self, nfe):
        start_date, end_date = '2022-07-01', '2023-07-10'
        nf_txn = nfe.filter(F.col('as_of_dt').between(start_date, end_date))
        exclude_cols = ['job_run_dt', 'dsf_data_origin_country', 'edsf_data_unit', 'batchid', 'job_nm']
        selected_cols = [col for col in nf_txn.columns if col not in exclude_cols]
        nf_txn = nf_txn.select(selected_cols)
        
        labels = self.generate_samples_labels(modelling_data=None) # load from parquet
        nf_txn = nf_txn.withColumn('date', F.date_add(F.col('as_of_dt'), 1))
        nf_txn_ft = labels.join(nf_txn, ['cif_id','date'])
        return nf_txn_ft
        
        
    
class DataProfiler:
    # ignore warning for using ragged tesnors
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    def __init__(self):
        self.reporting_columns = ['feature_name', 'records_count',
                                  'dtype','unique_vals',
                                  'count_missing_values', 'count_duplicated',
                                  'sampel_values', 'range_of_features']
    def get_numerical_features(self, df):
        numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
        return numerical_features
    
    def get_categorical_features(self, df):
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        return categorical_features
    
    def count_unique_values(self, df, col):
        return len(df[col].unique())
    
    def sample_unique_values(self, df, col, max_size=4):
        return df[col].unique()[:max_size]
    
    def count_missing_values(self, df, col):
        return df[col].isnull().sum()
    
    def get_range_of_values(self, df, col):
        if df[col].dtype == 'object':
            return f'min -> max: ({df[col].min()}, {df[col].max()}))'
        return f'min -> max: ({df[col].min():.3f, df[col].max():.3f})'
    
    def count_duplicated_values(self, df, col):
        return df[col].duplicated().sum()
    
    def is_date_field(self, df, col):
        if df[col].dtype =='object':
            try:
                x = pd.to_datetime(df[col])
                return True
            except (ValueError, OverflowError, TypeError) as e:
                return False
        return False
    
    def analyze_numerical_features(self, df, num_cols, profile_df):
        metadata_dict = defaultdict(lambda : {})
        for col in num_cols:
            # find total number of records
            metadata_dict[col]['num_records'] = df.shape[0]
            
            # find dtype
            metadata_dict[col]['dtype'] = df[col].dtype.name
            
            # find count unique values
            metadata_dict[col]['cnt_uv'] = self.count_unique_values(df, col)
            
            # count missing values
            metadata_dict[col]['cnt_miss'] = self.count_missing_values(df, col)
            
            # count duplicated values
            metadata_dict[col]['cnt_dup'] = self.count_duplicated_values(df, col)
            
            # sample values
            metadata_dict[col]['cnt_dup'] = self.sample_unique_values(df, col)
            
            # get range of data
            metadata_dict[col]['range_of_data'] = self.get_range_of_values(df, col)
            
            profile_df.loc[profile_df.shape[0], :] = [col] + list(metadata_dict[col].values())
        return profile_df
    
    def analyze_categorical_features(self, df, num_cols, profile_df):
        metadata_dict = defaultdict(lambda: {})
        for col in num_cols:
            # find total number of records
            metadata_dict[col]['num_records'] = df.shape[0]
            # find dtype
            metadata_dict[col]['dtype'] = df[col].dtype.name
            # find count unique values
            metadata_dict[col]['cnt_uv'] = self.count_unique_values(df, col)
            # find count missing values
            metadata_dict[col]['cnt_miss'] = self.count_missing_values(df, col)
            # find count duplicated values
            metadata_dict[col]['cnt_dup'] = self.count_duplicated_values(df, col)
            # sample values
            metadata_dict[col]['sample_val'] = self.sample_unique_values(df, col)
            # get range of data
            metadata_dict[col]['range_of_data'] = 'not_applicable' if not self.is_date_field(df, col) \
                else self.get_range_of_values(df[~df[col].isnull()], col)
            
            profile_df.loc[profile_df.shape[0], :] = [col] + list[metadata_dict[col].values()]
        return profile_df
    
    def add_original_spark_type(self, df, col_dtype):
        og_val = []
        for name in df.feature_name:
            og_val.append(col_dtype[name])
        df['spark_dtype'] = og_val
        return df
    
    def get_data_profile(self, df):
        """Get pandas dataframe containing data quality reports"""
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        df = df.copy()
        profiling_df = pd.DataFrame(columns=self.reporting_columns)
        
        # analyze numerical feaures
        nf = self.get_numerical_features(df)
        profiling_df = self.analyze_numerical_features(df, nf, profiling_df)
        
        # analyze categorical features
        cf = self.get_categorical_features(df)
        profiling_df = self.analyze_categorical_features(df, cf, profiling_df)
        profiling_df = profiling_df.sort_values('feature_name', ignore_index=True)
        return profiling_df

    def profile_data_sample(self, df, partition_col=None, partition_start_date=None, partition_end_date=None, sample_size=5000):
        col_dtype = dict(df.types)
        if partition_col and partition_start_date and partition_end_date:
            # remmeber to cast date time to stirng before this
            df_sample = df.filter(F.col(partition_col).between(partition_start_date, partition_end_date)).toPandas()
            df_profile = self.get_data_profile(df_sample.limit(sample_size))
            min_date = str(df.select(F.min(F.col(partition_col)).alias('min_dt')).take(2)[0].min_dt)
            max_date = str(df.select(F.max(F.col(partition_col)).alias('max_dt')).take(2)[0].max_dt)
            df_profile[f'min_{partition_col}'] = min_date
            df_profile[f'max_{partition_col}'] = max_date
        else:
            min_date = str(df.select(F.min(F.col(partition_col)).alias('min_dt')).take(2)[0].min_dt)
            max_date = str(df.select(F.max(F.col(partition_col)).alias('max_dt')).take(2)[0].max_dt)
            
            df_sample = df.limit(sample_size).toPandas()
            df_profile = self.get_data_profile(df_sample)
            df_profile[f'min_{partition_col}'] = min_date
            df_profile[f'max_{partition_col}'] = max_date
        df_profile['explored_date'] = datetime.datetime.now().strftime('%Y-%m-%d')
        df_profile['spark_dtype'] = df_profile['feature_name'].map(col_dtype)
        return df_profile


class ArtefactHandler:
    def __init__(self, name='default'):
        self.name = name
        self.artefact_db = {}
        is_trained=False
    def add_artefact(self, artefact, name):
        self.artefact_db[name] = artefact
        return
    def update_dict_artefact(self, name, key, value):
        artefact_to_update = self.get_artefact(name)
        artefact_to_update[key] = value
        self.artefact_db[name] = artefact_to_update
        return
    def get_artefact(self, name):
        return self.artefact_db[name]
    
    def return_all_artefacts(self):
        return self.artefact_db
    
    def save_to_pickle(self, file_name):
        artefact = self.return_all_artefacts()
        with open(f'{file_name}', 'wb') as handle:
            pickle.dump(artefact, handle)
        print(f'ArtefactHandler.save_to_pickle: artefacts saved to: {file_name}')
        return
    
    def load_from_pickle(self, file_name):
        with open(f'{file_name}', 'rb') as handle:
            artefact_db = pickle.load(handle)
        self.artefact_db = artefact_db
        self.is_trained=True
        print('ArtefactHandler.load_from_pickle: artefacts loaded from: {file_name}')
        return self
        

class FeatureAnalysis:
    def __init__(self,
                 cust_key = 'cif_id', 
                 pkeys=['cif_id','user_id','date','next_7_day_labels','cif', 'kais','party_id', 'customer_id', 'CBIN_CUST_ID'],
                 dkeys = ['date','month','week','as_of_dt'],
                 label = 'label',
                 drop_cols = ['col'],
                 date_col ='date',
                 numerical_missing_values= 0,
                 categorical_missing_value = 'NAN',
                 verbose=True,
                 ):
        self.date_col = date_col
        self.cust_key = cust_key
        self.pkeys = pkeys
        self.dkeys = dkeys
        self.label = label
        
        self.drop_cols = drop_cols
        self.df_pd = None
        
        # Store data summary from all the columns
        self.summary = None
        
        # store the features used
        self.features = None
        # store list of features with rule based issues
        self.all_missing_columns = None
        self.all_zero_columns = None
        self.all_single_cardinality_columns = None
        # store valid features based on summary
        self.valid_features = None
        # summary of only the valid features
        self.feature_summary = None
        # Store the numerical and categorical features
        self.numerical_columns = None
        self.categorical_columns = None
        self.numerical_missing_value = numerical_missing_values
        self.categorical_missing_value = categorical_missing_value
        
        # Store dictionary mapping from labels to idx and vice-versa
        self.lab_to_idx = None
        self.idx_to_lab = None
        # store the model
        self.lgbm = None
        
        self.verbose = verbose
        self.label_distribution = None
        
    def convert_spark_to_pandas(self, df_spark):
        self.df_pd = toPandas(df_spark)
        return self.df_pd
    
    def get_date_cols(self, df_pd, date_col='date'):
        if self.verbose:
            print('Creating additional `month` and `week` columns from date_col')
        og_pd = df_pd.copy()
        try:
            df_pd = og_pd.copy()
            df_pd = convert_day_to_month(df_pd, date_col)
            df_pd = convert_day_to_week(df_pd, date_col)
            df_pd = df_pd.sort_values(date_col)
            return df_pd
        except:
            df_pd = og_pd.copy()
            print('Not enough dates to create a `month` column')
            df_pd = convert_day_to_week(df_pd, date_col)
            df_pd = df_pd.sort_values(date_col)
            return df_pd
        
    def drop_columns(self, df_pd, cols):
        if self.verbose:
            print('Dropping adhoc columns from user import `drop_cols`: {cols}')
        return df_pd.drop(columns=cols, axis=1)
    
    def get_data_sumamry(self, df_pd):
        if self.verbose:
            print('Creating the summary of data. For example missing rate, zero rate')
        self.summary = get_data_summary(df_pd)
        return self.summary
    
    def get_feature_cols(self, df_pd):
        if self.verbose:
            print('Selecting the input features that are not (1) primary keys (2) date keys or (3) the label')
        self.features = [col for col in df_pd.columns if col not in self.pkeys + self.dkeys + [self.label]]
        return self.features
    
    def get_valid_features_from_summary(self, summary, df_pd, original_features, label, verbose=True):
        if self.verbose:
            print('Selecting the feasture that have (1) Missing rate != 1 (2) zero_rate != 1 and (3) Cardinality != 1')
        self.all_missing_columns = summary[summary['missing_rate']==1]['var'].unique().tolist()
        self.all_zero_columns = summary[summary['zero_rate']==1]['var'].unique().tolist()
        self.all_single_cardinality_columns = summary[summary['num_unique']==1]['var'].unique().tolist()
        
        data_summary_drop_cols = list(set(self.all_missing_columns + self.all_zero_columns + self.all_single_cardinality_columns))
        df_pd = df_pd[[col for col in df_pd[original_features + [label]].columns if col not in data_summary_drop_cols]]
        features = [col for col in df_pd if col in original_features]
        self.valid_features = features
        return self.valid_features

    def get_categorical_features_from_summary(self, feature_summary):
        if self.verbose:
            print('Selecting categorical features')
        categorical_columns = feature_summary[feature_summary['proposed_data_type']=='categorical']['var'].unique().tolist()
        return categorical_columns
    
    def get_numerical_features_from_summary(self, feature_summary):
        if self.verbose:
            print('Selecting numerical features')
        numerical_columns = feature_summary[feature_summary['proposed_data_type']=='numerical']['var'].unique().tolist()
        return numerical_columns
    
    def fill_missing_numerical_columns(self, df_pd, numerical_columns, fill_value=0):
        if self.verbose:
            print('Filling missing values in numeircal_column with value : {fill_value}')
        df_pd = df_pd.copy()
        for num in numerical_columns:
            df_pd[num] = df_pd[num].fillna(fill_value)
        return df_pd
    
    def fill_missing_categorical_columns(self, df_pd, categorical_columns, fill_value ='NAN'):
        if self.verbose:
            print('Filling missing values in categorical_columns with value: {fill_value}')
        df_pd = df_pd.copy()
        for cat in categorical_columns:
            df_pd = df_pd[cat].fillna(fill_value)
        return df_pd
    
    def cast_decimals_from_summary(self, df_pd, feature_summary, numerical_columns):
        if self.verbose:
            print(f'Casting decimal data types in numerical_columns to float')
        df_pd = df_pd.copy()
        # cast to decimal
        for col in tqdm.tqdm(numerical_columns):
            if decimal.Decimal in feature_summary.loc[feature_summary['var']==col]['type'].iloc[0]:
                df_pd[col] = df_pd[col].apply(lambda x: float(x) if (str(x)!='np.nan' and x!=None) else x)
        return df_pd
    
    def label_encode(self, df_pd, label):
        if self.verbose:
            print('Encoding labels(str) to index(int)')
        df_pd = df_pd.copy()
        lab_to_idx = {lab:i for i, lab in enumerate(df_pd[label].unique())}
        idx_to_lab = {i:lab for lab,i in lab_to_idx.items()}
        df_pd['label'] = df_pd[label].map(lab_to_idx)
        
        self.lab_to_idx= lab_to_idx
        self.idx_to_lab = idx_to_lab
        return df_pd
   
    def encode_categorical_columns(self, df_pd, categorical_columns):
        if self.verbose:
            print(f'Encoding categorical columns(str) to index(int)')
        
        df_pd = df_pd.copy()
        
        def encode_categorical(df_pd, categorical_columns):
            for cat in tqdm.tqdm(categorical_columns):
                le = preprocessing.LabelEncoder()
                df_pd[cat] = le.fit_transform(df_pd[cat])
            return df_pd

        df_pd = encode_categorical(df_pd, categorical_columns)
        return df_pd
                    
    def fill_missing_labels(self, y_train, y_test):
        
        missing_labels_in_y_train = [col for col in y_test.unique() if col not in y_train.unique()]
        missing_labels_in_y_test = [col for col in y_train.unique() if col not in y_test.unique()]
        print(f'missing_labels_in_y_train: {missing_labels_in_y_train}')
        print(f'missing_labels_in_y_test: {missing_labels_in_y_test}')
        
        
        # replace last few rows with the missing labels.
        # Note tha tis last few row contain a label that appeared only once, that removed label will become missing now
        for i in range(len(missing_labels_in_y_train)):
            y_train.iloc[-(i+1)] = missing_labels_in_y_train[i]
        
        for i in range(len(missing_labels_in_y_test)):
            y_test.iloc[-(i+1)] = missing_labels_in_y_test[i]
        return y_train, y_test
    
    def get_value_counts(self, y):
        label_name = y.name
        y_dist_perc = y.value_counts(normalize=True).reset_index().rename(columns={'index': label_name, label_name:'percentage'})
        y_dist_cnt = y.value_counts(normalize=False).reset_index().rename(columns={'index': label_name, label_name:'count'})
        return y_dist_cnt.merge(y_dist_perc, on=[self.label])
    
    def calculate_distribution(self, y, y_train, y_test):
        label_name = y.name
        y_dist = self.get_value_counts(y)
        y_dist[f'{label_name}_desc'] = y_dist[label_name].map(self.idx_to_lab)
        y_train_dist = self.get_value_counts(y_train).set_index(label_name).add_suffix('_train').reset_index()
        y_test_dist = self.get_value_counts(y_test).set_index(label_name).add_suffix('_test').reset_index()
        
        all_dist = y_dist.merge(y_train_dist, on=[label_name], how='left').merge(y_test_dist, on=[label_name], how='left')
        all_dist = all_dist.sort_values(by=['count'], ascending=False)
        
        order1 = [label_name, f'{label_name}_desc']
        order = order1 + [col for col in all_dist.columns if col not in order1]
        return all_dist[order]
    
    def get_train_test_counts(self, X, X_train_df, X_test_df, cust_key):
        counts = [X.shape[0], X_train_df.shape[0], X_test_df.shape[0]]
        unique_counts = [X[cust_key].unique(), X_train_df[cust_key].unique(), X_test_df[cust_key].unique()]
        df = pd.DataFrame({'dataset': ['original', 'train', 'test'],
                           'counts': counts,
                           f'unique_{cust_key}': unique_counts})
        return df
    
    def create_train_test_split(self, df_pd, cust_key, label, test_size=0.2, random_state=33):
        if self.verbose:
            print('Creating train test split for modelling')
        
        X, y = df_pd, df_pd[label]
        X_cin = pd.DataFrame(X[cust_key].unique(), columns=[cust_key])
        
        X_train_cin, X_test_cin = train_test_split(X_cin, test_size=test_size, random_state=random_state)
        
        # Get train test
        X_train_df = X_train_cin.merge(X, on=[cust_key])
        input_features = self.valid_features
        X_train = X_train_df[input_features]
        y_train = X_train_df[label]
        
        # Get test
        X_test_df = X_test_cin.merge(X, on=[cust_key])
        X_test = X_test_df[input_features]
        y_test = X_test_df[label]
        
        self.label_distribution = self.calculate_distribution(y, y_train, y_test)
        self.count_stats = self.get_train_test_counts(X, X_train, X_test)
        return X_train, X_test, y_train, y_test
    
    def initialize_lgbm(self, **kwargs):
        LGBM = lambda x: x
        return LGBM(features=self.valid_features,
                    objective='multiclassova',
                    num_class=len(self.lab_to_idx),
                    metric='auc_mu',
                    **kwargs)
    
        
    def run_data_pipeline(self, df_spark):
        df_pd = self.convert_spark_to_pandas(df_spark=df_spark)
        df_pd = self.get_date_cols(df_pd=df_pd, date_col = self.date_col)
        
        if len(self.drop_cols)>0:
            df_pd = self.drop_columns(df_pd, self.drop_cols)
        
        summary = self.get_data_summary(df_pd)
        features = self.get_feature_cols(df_pd)
        features = self.get_valid_features_from_summary(summary=summary,
                                                        df_pd=df_pd,
                                                        original_features=features,
                                                        label=self.label)
        self.feature_summary = summary[summary['var'].apply(lambda x: x in features)]
        self.categorical_columns = self.get_categorical_features_from_summary(self.feature_summary)
        self.numerical_columns = self.get_numerical_features_from_summary(self.feature_summary)
        
        df_pd = self.fill_missing_numerical_columns(df_pd, self.numerical_columns, fill_value=self.numerical_missing_value)
        df_pd = self.fill_missing_categorical_coluns(df_pd, self.categorical_columns, fill_value=self.categorical_missing_value)
        df_pd = self.cast_decimals_from_summary(df_pd, self.feature_summary, self.numerical_columns)
        df_pd = self.encode_categorical_columns(df_pd, self.categorical_columns)
        return df_pd
    
    def run_model_pipeline(self, df_pd, **kwargs):
        df_pd = self.encode_label(df_pd, label=self.label)
        X_train, X_test, y_train, y_test = self.create_train_test_split(df_pd, self.cust_key, self.label, test_size=0.2, random_state=33)
        y_train, y_test = self.fill_missing_labels(y_train, y_test)
        self.lgbm = self.initialize_lgbm(**kwargs)
       
        self.lgbm.fit(X=X_train, y=y_train, xeval=X_test, yeval=y_test, categorical_features=self.categorical_columns) 
        
        # plot learning curbe
        self.lgbm.plot_learning_curve()
        self.fi = self.lgbm.get_feature_importance_elbow_plot()
        return self.lgbm
    
    def gen_feature_report(self, number_of_features):
        feature_importance_df = self.fi
        feature_importance_df = feature_importance_df.reset_index(drop=True).reset_index()
        feature_importance_df['is_model_selected_flag'] = feature_importance_df['index'].apply(lambda x: 1 if x < number_of_features else 0)
        feature_importance_df =  feature_importance_df[['feature','gain_importance','split_importance', 'is_model_selected_flag']]
        
        original_features = self.summary[['var']].copy()
        feature_selection_report = original_features.merge(feature_importance_df,
                                                           left_on='var',
                                                           right_on='feature',
                                                           how='left')
        feature_selection_report['used_as_model_input'] = feature_selection_report['gain_importance'].apply(lambda x: 1 if str(x)!='nan' else 0)
        feature_selection_report['remarks'] = ''
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, missing_rate==1" \
            if x['var'] in self.all_missing_columns else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, zero_rate==1" \
            if x['var'] in self.all_zero_columns else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, cardinality==1" \
            if x['var'] in self.all_single_cardinality_columns else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, is model selected" \
            if x['is_model_selected_flag']==1 else \
                f"{x['remarks']}, is not model selected" if x['is_model_selected_flag']==0 else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, is primary key" \
            if x['var'] in self.pkeys else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"{x['remarks']}, is date key" \
            if x['var'] in self.dkeys else x['remarks'], axis=1)
        feature_selection_report['remarks'] = feature_selection_report.apply(lambda x: f"x['remarks], is label" \
            if x['var']==self.label else x['remarks'], axis=1)
        feature_selection_report = feature_selection_report.sort_values('gain_importance', asending=False)
        feature_selection_report = feature_selection_report[['var','used_as_model_input','is_model_selected_flag', 'remarks', 'gain_importance', 'split_importance']].fillna('NA:dropped_from_model_input')
        feature_selection_report = feature_selection_report.reset_index(drop=True)
        return feature_selection_report
    
    def save_feature_analysis_report(self, writer, num_features):
        # Save summary
        writer.write_to_excel('data_summary', self.summary)
        # Save train test count
        writer.write_to_excel('train_test_count', self.count_stats)
        # Save label distribuion
        writer.write_to_excel('label_dist', self.label_distribution)
        # Save learning curbe data
        writer.write_to_excel('learning_details', self.lgbm.learning_df)
        # Save feature importance
        writer.write_to_excel('feature_importance', self.fi)
        # Save feature report
        fa_rep = self.gen_feature_report(number_of_features= num_features)
        writer.write_to_excel('feature_selection_report', fa_rep)
        return fa_rep

    def main(self):
        demo = None
        fa = FeatureAnalysis(demo)
        df_pd = fa.run_data_pipeline()
        lgbm = fa.run_model_pipeline(df_pd, max_depth=3, lambda_l2 = 10, lambda_l1 = 10)
        lgbm.plot_learning_curve()
        fi = lgbm.get_feature_importance_elbow_plot()
        fa.gen_feature_report(50)
        
        writer= ExcelWriter(base_folder='data/samples/', filename='MVP2_demo_48_intents')
        fa.save_feature_analysis_report(writer, num_features=30)
        return
        
            
    def label_encode_train(self, train, label):
        if self.verbose:
            print('Encoding labels(str) to index(int)')
        train = train.copy()
        y_train = train[label]
        lab_to_idx = {lab:i for i, lab in enumerate(y_train.unique())}
        idx_to_lab = {i:lab for lab,i in lab_to_idx.items()}
        train['label'] = y_train[label].map(lab_to_idx)
        
        self.lab_to_idx= lab_to_idx
        self.idx_to_lab = idx_to_lab
        
        # Add label encoder to 'training artefact'
        label_encoder_artefacts = {'label_encoders': {'lab_to_idx': lab_to_idx,
                                                      'idx_to_lab': idx_to_lab}}
        
        self.artefact_handler.add_artefact(label_encoder_artefacts, name='training')
        return train
    
    def label_encode_score(self, score, label):
        y_score = score[label].copy()
        
        # map using label mapping from training
        lab_to_idx = self.artefact_handler.get_artefact('training')['label_encoders']['lab_to_idx']
        score[label] = score[label].map(lab_to_idx)
        if score[label].isnull().sum() > 0:
            labels_not_found = [lab for lab in y_score.unique() if lab not in lab_to_idx.keys()]
            print(f'Label score not found in encoders in training: {labels_not_found}')
        # drop those labels without map
        score = score.dropna(subset=[label])
        score[label] = score[label].astype(int)
        return score
        
    def label_encode_retrain(self, retrain, label):
        y_retrain = retrain[label]
        
        # Get label mapping from training data
        lab_to_idx = {lab: i for i, lab in enumerate(y_retrain.unique())}
        idx_to_lab = {i:lab for lab, i in lab_to_idx.items()}
        retrain[label] = y_retrain.map(lab_to_idx)
        # add the label encoders to 'retraining artefacts'
        label_encoder_artefacts = {'label_encoders': {'lab_to_idx': lab_to_idx,
                                                      'idx_to_lab': idx_to_lab}}
        self.artefact_handler.add_artefact(label_encoder_artefacts, name='retraining')
        return retrain
    
    def train_test_split(self, df_pd):
        X = df_pd.copy()
        
        X_cin = pd.DataFrame(X['cif_id'].unique(), coluns=['cif_id'])
        # divide the unique customers into training and testing
        X_train_cin, X_test_cin = train_test_split(X_cin, test_size=float(0.2), random_state= int(99))
        
        # select data that contains customers in training
        X_train_df = X_train_cin.merge(X, on=['cif_id'])
        X_test_df = X_test_cin.merge(X, on=['cif_id'])
        
        train = X_train_df
        test = X_test_df
        return train, test
        
    
    def get_train_test_oos_for_training(self, df_pd):
        train_start, train_end = '2022-08-10', ' 2023-03-10'
        test_start, test_end = '2023-03-10', '2024-05-10'
        # Get data belonging to train and test
        df_pd_train_test = df_pd[df_pd['date'].apply(lambda x: (x>=train_start) and (x<train_end))]
        train, test = self.train_test_split(df_pd_train_test)

        # get OOS dataset
        oos_df = df_pd[df_pd['date'].apply(lambda x: (x>=test_start) & (x<=test_end))]
        
        # Label Encode Data
        train = self.label_encode_train(train)
        # Get label encoding from scoring
        test = self.label_encode_score(test)
        oos_df = self.label_encode_score(oos_df)
        return train, test, oos_df
    
    def encode_categorical_train(self, df_pd, categorical_columns):
        print('Encoding Categorical Features for training.')
        df_pd = df_pd.copy()
        categorical_feature_maps = {}
        for cat in tqdm.tqdm(categorical_columns):
            # Get distribution in descending order
            unique_vals_dist = df_pd[cat].value_counts().reset_index().values
            # Set the value with least percentage as 'least_percentage'
            sorted_unique_vals = [v[0] for v in unique_vals_dist]
            sorted_unique_vals = sorted_unique_vals[:-1] + ['least_perc']
            cat_map = {val:idx for idx, val in enumerate(sorted_unique_vals)} 
            # map values and set the value with least percentage, and new values as 'least_perc'
            df_pd[cat] = df_pd[cat].map(cat_map).fillna(cat_map['least_perc'])
            # save encoding for cat feature
            categorical_feature_maps[cat] = cat_map
        self.artefact_handler.update_dict_artefact(name='training', key='categorical_feature_maps', value=categorical_feature_maps)
        return df_pd
    
    def encode_categorical_score(self, df_pd, categorical_columns):
        df_pd = df_pd.copy()
        categorical_feature_maps = self.artefact_handler.get_artefact('training')['categorical_feature_maps']
        for cat in tqdm.tqdm(categorical_columns):
            cat_map = categorical_feature_maps[cat]
            df_pd[cat] = df_pd[cat].map(cat_map).fillna(cat_map['least_perc'])
        return df_pd
    
    def encode_categorical_retrain(self, df_pd, categorical_columns):
        print('Encoding Categorical Features for training.')
        df_pd = df_pd.copy()
        categorical_feature_maps = {}
        for cat in tqdm.tqdm(categorical_columns):
            # Get distribution in descending order
            unique_vals_dist = df_pd[cat].value_counts().reset_index().values
            # Set the value with least percentage as 'least_percentage'
            sorted_unique_vals = [v[0] for v in unique_vals_dist]
            sorted_unique_vals = sorted_unique_vals[:-1] + ['least_perc']
            cat_map = {val:idx for idx, val in enumerate(sorted_unique_vals)} 
            # map values and set the value with least percentage, and new values as 'least_perc'
            df_pd[cat] = df_pd[cat].map(cat_map).fillna(cat_map['least_perc'])
            # save encoding for cat feature
            categorical_feature_maps[cat] = cat_map
        self.artefact_handler.update_dict_artefact(name='retraining', key='categorical_feature_maps', value=categorical_feature_maps)
        return df_pd

    def encode_categorical_retrain_score(self, df_pd, categorical_columns):
        df_pd = df_pd.copy()
        categorical_feature_maps = self.artefact_handler.get_artefact('retraining')['categorical_feature_maps']
        for cat in tqdm.tqdm(categorical_columns):
            cat_map = categorical_feature_maps[cat]
            df_pd[cat] = df_pd[cat].map(cat_map).fillna(cat_map['least_perc'])
        return df_pd
    
            
        
        
            
            
        
    

    
            