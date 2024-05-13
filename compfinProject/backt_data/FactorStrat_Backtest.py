#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.api import OLS


# In[2]:


stock1 = pd.read_csv('TRD_BwardQuotation17_0.csv')
stock2 = pd.read_csv('TRD_BwardQuotation17_1.csv')
stock3 = pd.read_csv('TRD_BwardQuotation17_2.csv')
stock4 = pd.read_csv('TRD_BwardQuotation18_0.csv')
stock5 = pd.read_csv('TRD_BwardQuotation18_1.csv')
stock6 = pd.read_csv('TRD_BwardQuotation18_2.csv')
stock7 = pd.read_csv('TRD_BwardQuotation18_3.csv')


# In[3]:


print(stock1['ShortName'].unique())
print(stock2['ShortName'].unique())
print(stock3['ShortName'].unique())
print(' ')
print(stock4['ShortName'].unique())
print(stock5['ShortName'].unique())
print(stock6['ShortName'].unique())
print(stock7['ShortName'].unique())


# In[4]:


combined = pd.concat([stock1,stock2,stock3]).drop_duplicates()
combined2 = pd.concat([stock4,stock5,stock6,stock7]).drop_duplicates()


# In[5]:


set1 = set(combined['ShortName'])
set2 = set(combined2['ShortName'])
common = set1.intersection(set2)
com1 = combined[combined['ShortName'].isin(common)]
com2 = combined2[combined2['ShortName'].isin(common)]
stockdata0 = pd.concat([com1,com2])
stockdata = stockdata0[['TradingDate','ShortName','ChangeRatio']]
stockdata_close = stockdata0[['TradingDate','ShortName','ClosePrice']]
stockdata_close


# In[6]:


stock1 = pd.concat([com1,com2])
stockdata_open = stock1[['TradingDate','ShortName','OpenPrice']]
stockdata_open


# In[7]:


stockdata.set_index('TradingDate',inplace=True)
stockdata = stockdata.pivot(columns='ShortName',values='ChangeRatio')
stockdata

stockdata0.set_index('TradingDate',inplace=True)
stockdata0 = stockdata0.pivot(columns='ShortName',values='ClosePrice')
stockdata0


# In[8]:


stockdata_open.set_index('TradingDate',inplace=True)
stockdata_open = stockdata_open.pivot(columns='ShortName',values='OpenPrice')
stockdata_open


# In[9]:


minimum = int(0.2*len(stockdata))
stockdata = stockdata.dropna(axis=1,thresh=minimum)
stockdata
# 扔掉缺失值超过80%的列


# In[10]:


stockdata_open = stockdata_open.dropna(axis=1,thresh=minimum)
stockdata_open


# In[11]:


col1 = [col for col in stockdata if 'ST' in col]
stockdata = stockdata.drop(columns=col1)
col1 = [col for col in stockdata0 if 'ST' in col]
stockdata0 = stockdata0.drop(columns=col1)


# In[12]:


col1 = [col for col in stockdata_open if 'ST' in col]
stockdata_open = stockdata_open.drop(columns=col1)
stockdata_open


# In[13]:


common_columns = stockdata.columns.intersection(stockdata0.columns)
stockdata = stockdata[common_columns]
stockdata0 = stockdata0[common_columns]


# In[14]:


stockdata_open = stockdata_open[common_columns]
stockdata_open


# In[15]:


Market_type = pd.read_csv('TRD_Co.csv')
Market_type = Market_type[['Stknme','ID']]

Market_type


# In[16]:


aa = pd.DataFrame(common_columns)
aa


# In[17]:


# 将Series转换为集合并找交集
common_elements = Market_type[['Stknme','ID']][Market_type['Stknme'].isin(set(aa['ShortName']))]
common_elements = common_elements[common_elements['ID']=='P9701']
common_elements


# In[18]:


company_names = common_elements['Stknme'].tolist()

filtered_stockdata_open = stockdata_open[company_names]
filtered_stockdata = stockdata[company_names]
filtered_stockdata0 = stockdata0[company_names]


# In[19]:


famafrench = pd.read_csv("STK_MKT_THRFACDAY_m.csv")
famafrench = famafrench[famafrench['MarkettypeID']=='P9701']
famafrench = famafrench[(famafrench['TradingDate']>='2014-01-01')&(famafrench['TradingDate']<='2021-04-30')]
famafrench


# In[20]:


number_of_stocks = stockdata.T.count()

number_of_stocks_toomuch = abs(stockdata) >= 0.09
number_of_stocks_toomuch = stockdata.T.sum()

factor = number_of_stocks_toomuch/ number_of_stocks
factor.replace(np.nan, 0, inplace = True)
factor = pd.DataFrame(factor)
factor = factor.rename(columns={0:'LimitHitters'})
factor


# In[21]:


famafrench['TradingDate'] = famafrench['TradingDate']
famafrench.set_index('TradingDate', inplace=True)
famafrench


# In[22]:


common_dates = factor.index.intersection(famafrench.index)
common_dates
factor = factor.loc[common_dates]
famafrench = famafrench.loc[common_dates]


# In[23]:


# famafrench = famafrench.reset_index(drop= True)
famafrench['TradingDate'] = factor.index
famafrench = famafrench[['TradingDate','RiskPremium2', "SMB2", "HML2"]]
famafrench


# In[24]:


stock_close_common0 = filtered_stockdata0 
stock_close_common0 = factor.merge(stockdata0,left_index=True,right_index=True,how='inner')
stock_close_common0 = stock_close_common0.drop(columns='LimitHitters')
stock_close_common0


# In[25]:


all_results = {}
# 计算全部股票收益率
daily_returns = stock_close_common0.pct_change()
daily_returns = daily_returns.loc['2014-01-02':'2021-04-30', :]
daily_returns.index = pd.to_datetime(daily_returns.index)
daily_returns


# In[26]:


factor.index
factor.index = pd.to_datetime(factor.index)
grouped = factor.groupby([factor.index.year, factor.index.month])
filtered_trading_days = pd.DataFrame({
    'start_date': [group[1].index.min() for group in grouped],
    'end_date': [group[1].index.max() for group in grouped]
})
filtered_trading_days.reset_index(drop=True, inplace=True)
print(filtered_trading_days)
#为了好分组


# In[27]:


filtered_stockdata_open = stockdata_open[company_names]


# In[28]:


stockdata_close_shift = stock_close_common0.shift(-1)
stockdata_close_shift.index = pd.to_datetime(stockdata_close_shift.index)
stockdata_close_shift = stockdata_close_shift[stockdata_close_shift.index.isin(filtered_trading_days.end_date)]
stockdata_close_shift = stockdata_close_shift.pct_change()
stockdata_close_shift


# In[29]:


stockdata_open = filtered_stockdata_open
stockdata_open_shift = stockdata_open.shift(-1)
stockdata_open_shift.index = pd.to_datetime(stockdata_open_shift.index)
stockdata_open_shift = stockdata_open_shift[stockdata_open_shift.index.isin(filtered_trading_days.end_date)]
stockdata_open_shift = stockdata_open_shift.pct_change()
stockdata_open_shift


# In[30]:


fourfactors = pd.merge(pd.DataFrame(factor).reset_index(drop=True), famafrench.reset_index(drop=True) , left_index=True, right_index=True, how='left')
fourfactors = fourfactors[['TradingDate','LimitHitters','RiskPremium2','SMB2','HML2']]
fourfactors = fourfactors.rename(columns={'TradingDate_x':'TradingDate', 'LimitHitters':'APL'})

fourfactors_reordered = fourfactors.set_index('TradingDate')
fourfactors_reordered


# In[31]:


fourfactors_reordered


# In[32]:


fourfactors_reordered.index = pd.to_datetime(fourfactors_reordered.index)
daily_returns.index = pd.to_datetime(daily_returns.index)


# In[33]:


for stock in daily_returns.columns:

    Y = daily_returns[stock]
    X = fourfactors_reordered
    X['RiskPremium2'] = pd.to_numeric(X['RiskPremium2'], errors='coerce')
    X['SMB2'] = pd.to_numeric(X['SMB2'], errors='coerce')
    X['HML2'] = pd.to_numeric(X['HML2'], errors='coerce')

    # match the index of X and Y
    Y = Y.loc[X.index]
    # print(Y)
    # print(X)

    stock_results = {}

    for index, row in filtered_trading_days.iterrows():

        start_date = pd.to_datetime(row['start_date'], format='%Y%m%d')
        end_date = pd.to_datetime(row['end_date'], format='%Y%m%d')
        X_subset = X.loc[(X.index >= start_date) & (X.index <= end_date)]
        Y_subset = Y.loc[(Y.index >= start_date) & (Y.index <= end_date)]

        if len(X_subset) == len(Y_subset) and len(X_subset) > 0:

            # handle missing values
            X_subset = X_subset.dropna()
            Y_subset = Y_subset.loc[X_subset.index]

            X_subset = X_subset.reset_index(drop=True)
            Y_subset = Y_subset.reset_index(drop=True)

            # Add a constant for the regression intercept
            X_with_const = sm.add_constant(X_subset, has_constant='add')


            # Run the regression
            model = sm.OLS(Y_subset, X_with_const)
            regression_results = model.fit()

            # Store the regression results for this month
            stock_results[end_date] = regression_results.params

        else:
            print('No data for stock {} for the period from {} to {}'.format(stock, start_date, end_date))
            stock_results[end_date] = np.nan

    # Store the results for the current stock
        all_results[stock] = stock_results
    # break

# Convert all_results to a DataFrame for a clean and organized output
all_params = {(outerKey, innerKey): values for outerKey, innerDict in all_results.items() for innerKey, values in innerDict.items()}
param_df = pd.DataFrame(all_params).T
param_df.index.names = ['Stock', 'End_Date']

# Display the organized results
print(param_df)


# In[ ]:


def align_factor(factors):
    '''
    将factors,按照aim_factor完成对齐,factors=[aim_factor,factor1,factor2]
    以第一个df为标准对齐
    需要考虑第一个factor有的日期,后面的factor可能会没有,所以,第一步应该是确保factor[1:]中的日期要有,没有的话向前填充
    '''
    # 先对齐日期
    index = factors[0].index
    columns = factors[0].columns
    result = [factors[0]]
    for i in factors[1:]:
        to_append = i.reindex(index=index,method='pad')
        to_append = to_append.reindex(columns=columns)
        result.append(to_append)
    return result


# In[ ]:


def FactorIC(factor1,factor2,min_valid_num=0):
    '''
    计算factor1与factor2的横截面相关系数(Pearson,Spearman)
    :param factor1(pd.DataFrame):因子1
    :param factor2(pd.DataFrame):因子2
    :param min_valid_num(float):横截面上计算一期相关系数最小的样本个数要求,默认最小是1
    :return pearson_corr,spearman_corr
    '''
    factor1.replace({None:np.nan},inplace=True)
    factor1=factor1.astype(float)
    factor2.replace({None:np.nan},inplace=True)
    factor2=factor2.astype(float)

    factor1_sum = factor1.notnull().sum(axis=1)
    factor1.loc[factor1_sum<min_valid_num,:]=np.nan
    factor2_sum = factor2.notnull().sum(axis=1)
    factor2.loc[factor2_sum<min_valid_num,:]=np.nan

    pearson_corr=factor1.corrwith(factor2,axis=1)
    spearman_corr=factor1.rank(axis=1).corrwith(factor2.rank(axis=1),axis=1)

    return pearson_corr,spearman_corr


# In[ ]:


def FactorGroup(factor,split_method='average',split_num=5,industry_factor=None,limit_df=None):
    '''
    将因子进行分类,按照行分类。
    :param factor:要分类的因子,或者打分
    :param split_method:默认为等比例分组,'average',还可以有'largest','smallest','largest_ratio','smallest_ratio'
    :param split_num:若split_method=='average',则等分split_num组,若为'largest',则最大n个,若'smallest',则最小n个,若largest_ratio,则最大百分比,若smallest_ratio,则最小百分比
    :param industry_factor(pd.DataFrame or None):行业因子
    :param limit_df(pd.DataFrame or NOne):None或者TrueFalse构成的df,来自于FactorTool_GetLimitDf的结果
    :return:factor_split_result, df
    '''
    if limit_df is not None:
        [factor,limit_df] = align_factor([factor,limit_df])
        limit_df = limit_df.fillna(value=True).astype('bool')
        factor = factor[limit_df]

    if industry_factor is None:
        industry_factor = pd.DataFrame(index=factor.index,columns=factor.columns,data='Market')
        industry_factor = industry_factor[factor.notnull()].astype('object')
    else:
        [factor,industry_factor] = align_factor([factor,industry_factor])
        industry_factor = industry_factor.astype('object')
        industry_factor = industry_factor.fillna(value='others')
        industry_factor = industry_factor[factor.notnull()]

    data = pd.DataFrame(index=pd.MultiIndex.from_product([factor.index,factor.columns],names=['date','asset']))
    data['group'] = industry_factor.stack()
    data['factor'] = factor.stack()
    data = data.dropna(subset=['group'])

    data_factor_array = data['factor'].values
    data_final_split = np.full((len(data_factor_array),),np.nan)

    grouper = [data.index.get_level_values('date'),'group']
    data_groupby = data.groupby(grouper)
    data_groupby_indices = data_groupby.indices
    data_groupby_indices = list(data_groupby_indices.values())

    def auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num):
        def quantile_split(_this_split_result,_this_array,_split_percentile):
            split_value = np.nanpercentile(_this_array,_split_percentile)
            split_value[0] -= 1
            split_value[-1] += 1
            for i in range(len(split_value)-1):
                _this_split_result[(_this_array<=split_value[i+1])&(_this_array>split_value[i])] = i
            return _this_split_result

        if split_method=='average':
            split_percentile = np.linspace(0,100,split_num+1)
        elif split_method=='largest_ratio':
            split_percentile = np.array([0,100-split_num*100,100])
        elif split_method=='smallest_ratio':
            split_percentile = np.array([0,split_num*100,100])

        for this_group_place in range(len(data_groupby_indices)):
            this_indice_place = data_groupby_indices[this_group_place]
            this_factor_array = data_factor_array[this_indice_place]
            this_split_result = data_final_split[this_indice_place]
            # if split_method in ['average','largest','smallest']:
            if split_method =='average':
                this_data_final_split = quantile_split(this_split_result,this_factor_array,split_percentile)
                data_final_split[this_indice_place] = this_data_final_split
            elif split_method=='smallest':
                this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])
                split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]
                if len(split_value)>0:
                    this_split_result[this_factor_array<=split_value]=0
                    this_split_result[this_factor_array>split_value]=1
                    data_final_split[this_indice_place] = this_split_result
            elif split_method=='largest':
                this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])[::-1]
                split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]
                if len(split_value)>0:
                    this_split_result[this_factor_array<split_value] = 0
                    this_split_result[this_factor_array>=split_value] = 1
                    data_final_split[this_indice_place] = this_split_result
        return data_final_split

    data_final_split = auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num)
    data.loc[:,'factor'] = data_final_split
    final_data = data['factor'].unstack().reindex(index=factor.index,columns=factor.columns)

    return final_data


# In[ ]:


def simple_factor_test(factor, use_data = 'this_close'):
    if use_data=='this_close':
        this_ret_data = stockdata_open_shift.shift(-1)
    else:
        this_ret_data = stockdata_open_shift.shift(-1)
    ic,rankic = FactorIC(factor,this_ret_data) 
    factor_group = FactorGroup(factor)
    condata = pd.concat([factor_group.unstack(),this_ret_data.unstack()],axis=1).dropna().reset_index()
    condata.columns =['stockcode','date','group_id','ret']
    group_ret = condata.groupby(['date','group_id'])['ret'].mean().unstack()
    return ic,rankic,group_ret,factor_group


# In[ ]:


pivoted_df = param_df["APL"].unstack().T
pivoted_df.index = pd.to_datetime(pivoted_df.index)
pivoted_df_monthly = pivoted_df.reindex(index = filtered_trading_days.end_date) 
pivoted_df_monthly


# In[ ]:


pivoted_df_monthly = abs(pivoted_df_monthly)
#pivoted_df_monthly = pivoted_df_monthly


# In[ ]:


pivoted_df_monthly


# In[ ]:


ic,rankic,group_ret,factor_group  = simple_factor_test(pivoted_df_monthly)


# In[ ]:


rank_ic_mean = ic.mean()
rank_ic_std = ic.std()
rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else float('nan')
ic_win_rate = (ic > 0).mean()

print(f"Rank IC Mean: {rank_ic_mean:.4f}")
print(f"Rank ICIR: {rank_icir:.4f}")
print(f"IC Win Rate: {ic_win_rate:.2%}")


# In[ ]:


group_ret


# In[ ]:


cumulative_returns = (1 + group_ret).cumprod() - 1

# 绘制累积收益率
plt.figure(figsize=(14, 7))
for column in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[column], label=f'Group {column}')

plt.title('Cumulative Group Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




