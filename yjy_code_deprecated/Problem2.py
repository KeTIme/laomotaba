#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:17:01 2019

@author: FedericoYe
"""

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
#------------------------------------------Preprocess data functions-----------------------------------------------------------------------------------


# This function is to convert tick datetime to readable date time
def convert_hepler(ticks):

    #date=datetime.datetime.fromtimestamp(ts).strftime("%B %d, %Y %I:%M:%S")
    date=datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(microseconds=ticks/10)
    
    return date

# This function is to convert tick datetime to readable date time
def date_conversion(ticks):
    #dateinsecond=dateinsecond.astype(float)
    dates = pd.Series(map(convert_hepler, ticks))
    dates.index=ticks.index
    
    return dates

# Loading data based on file path
# At the same time, filtering out trade Volume =0
def loadRawData(aPath,aFilePrefix,aDate):
    fullname=aPath + aFilePrefix + '_' + str(aDate)+'.csv'
    
    
    data=pd.read_csv(fullname,dtype={'RecvTime':'Int64','ExchTime':'Int64',
                                        'Bid': 'float', 'Ask': 'float', 'Last': 'float',
                                     'BidVol': 'Int64', 'AskVol': 'Int64', 'LastVol': 'Int64'})
    
    data=data[(data['BidVol']!=0)]
    data=data[(data['AskVol']!=0)]
    
    return data

#Function to readin raw data, get specific contract data within a time interval
def filter_data(aPath,aFilePrefix,aDate,ContractID):
    
    rawdata=loadRawData(aPath,aFilePrefix,aDate)
    newData=rawdata[rawdata['ContractId'].isin(ContractID)]
    newdate=date_conversion(newData['RecvTime'])
    
    Year=int(str(aDate)[:4])
    Month=int(str(aDate)[4:6])
    Day=int(str(aDate)[6:])
    
    #I set starting time at 9:05 since some of the contracts do not have tick data at 9:15
    #We can filter out data between 9:05 to 9:15 later
    starting_date_a=datetime.datetime(Year,Month,Day,9,5)
    end_date_a=datetime.datetime(Year,Month,Day,11,30)
    
    #Same Logic for the afternoon trading time
    starting_date_p=datetime.datetime(Year,Month,Day,13,20)
    end_date_p=datetime.datetime(Year,Month,Day,15,00)
    
    selected_time=newdate[((newdate > starting_date_a) & (newdate < end_date_a))|
          ((newdate > starting_date_p) & (newdate < end_date_p))]
    
    
    newData=newData.loc[selected_time.index]
    newData['time']=selected_time
    
    cleanData=newData.drop(columns=['RecvTime','ExchTime'])
    cleanData=cleanData.sort_values(by=['time'])
    
    return cleanData


# This function is to align price data.
#Since some of the contracts are illiquid, there are only few tick price data in a day.
#Thus, my approach is to restructure data in seconds. At the same time, this function 
# will only get data between 9:15 to 11:30 and 1:30 and 3:00.
def restructure_all_price_data(all_new_data,aDate,day_num,ContractID):
    
    Year=int(str(aDate)[:4])
    Month=int(str(aDate)[4:6])
    Day=int(str(aDate)[6:])
    #generate standard time stamp
    times_a = [(datetime.datetime(Year, Month, Day, 9, 0) + datetime.timedelta(seconds=x)) for x in range(145*60)]
    times_p = [(datetime.datetime(Year, Month, Day, 13, 20) + datetime.timedelta(seconds=x)) for x in range(100*60)]
    full_timestamp=times_a+times_p
    
    
    firstday=all_new_data[day_num]
    
    firstday['time']=firstday['time'].dt.round('1s')
    firstday=firstday.set_index('time')
    
    firstday['Mid']=firstday[['Bid','Ask']].mean(axis=1)
    
    
    starting_date_a=datetime.datetime(Year,Month,Day,9,14,59)
    end_date_a=datetime.datetime(Year,Month,Day,11,30)
    
    starting_date_p=datetime.datetime(Year,Month,Day,13,29,59)
    end_date_p=datetime.datetime(Year,Month,Day,15,00)
    
    allContract_price_last=pd.DataFrame()
    allContract_price_mid=pd.DataFrame()

    allContract_fulldata=pd.DataFrame()
    
    
    for i in range(len(ContractID)):
        
        current_contract=pd.DataFrame(index=full_timestamp)
        priceData=firstday[firstday['ContractId']==ContractID[i]][['PartitionDay','ContractId','Mid','BidVol','AskVol','Last']]
        priceData= priceData.loc[~priceData.index.duplicated(keep='last')]
        #allContract_price[str(ContractID[i])]=priceData['Mid']
        current_contract[['PartitionDay','ContractId','Mid','BidVol','AskVol','Last']]=priceData[['PartitionDay','ContractId','Mid','BidVol','AskVol','Last']]
        
        current_contract=current_contract.fillna(method='ffill')
        
        current_contract_clean=current_contract[((current_contract.index > starting_date_a) & (current_contract.index < end_date_a))|
          ((current_contract.index > starting_date_p) & (current_contract.index < end_date_p))]
        
        allContract_price_last[(ContractID[i])]=current_contract_clean['Last'] #['Mid']
        allContract_price_mid[(ContractID[i])]=current_contract_clean['Mid']
        
        #allContract_fulldata=pd.concat([allContract_fulldata,current_contract])
        
        current_contract_clean['ContractId'] = current_contract_clean['ContractId'].astype(int)
        current_contract_clean['PartitionDay'] = current_contract_clean['PartitionDay'].astype(int)
        
        #allContract_fulldata[str(ContractID[i])]=current_contract_clean
        allContract_fulldata=pd.concat([allContract_fulldata,current_contract_clean])
        
    
    #allContract_firstday=allContract_firstday.fillna(method='ffill')
    
    
    
#    filterData_allContract=allContract_price[((allContract_price.index > starting_date_a) & (allContract_price.index < end_date_a))|
#          ((allContract_price.index > starting_date_p) & (allContract_price.index < end_date_p))]
#    
#    firstday_filter=allContract_fulldata[((allContract_fulldata.index > starting_date_a) & (allContract_fulldata.index < end_date_a))|
#          ((allContract_fulldata.index > starting_date_p) & (allContract_fulldata.index < end_date_p))]
#    
    
    return allContract_price_last,allContract_price_mid,allContract_fulldata


#This Function serves to plot the movement of price within a time interval
#It also will plot the ratio of price of two securities to check if they are stationary.
# (Ploting the ratio of two prices, although is not a very formal way, but a good way
#to do sanity check)
def plot_spread(df, tick1, tick2, index, op, stop):
  
    px1 = df[tick1].iloc[index] / df[tick1].iloc[index[0]]
    px2 = df[tick2].iloc[index] / df[tick2].iloc[index[0]]
    
    px1=np.log(px1)
    px2=np.log(px2)
    
    sns.set(style='white')
    
    # Set plotting figure
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    # Plot the return plot
    sns.lineplot(data=[px1, px2], linewidth=1.2, ax=ax[0])
    ax[0].legend(loc='upper left')
    
    # Calculate the spread and other thresholds
    spread = df[tick1].iloc[index] - df[tick2].iloc[index]
    #spread = px1 - px2

    #spread = np.log(df[ticker1].iloc[idx]) - np.log(df[ticker2].iloc[idx])
    mean = spread.mean()
    sell_th     = mean + op
    sell_stop   = mean + stop
    buy_th      = mean - op
    buy_stop    = mean - stop
    
    # Plot the 2nd subplot
    sns.lineplot(data=spread, color='#85929E', ax=ax[1], linewidth=1.2)
    ax[1].axhline(sell_th,   color='b', ls='--',  label='sell_threadshold',linewidth=2)
    ax[1].axhline(buy_th,    color='r', ls='--', label='buy_threadshold',linewidth=2)
    ax[1].axhline(sell_stop, color='g', ls='--',  label='sell_stop',linewidth=1)
    ax[1].axhline(buy_stop,  color='y', ls='--',  label='buy_stop',linewidth=1)
    
    
    ax[1].fill_between(idx, sell_th, buy_th, facecolors='r',alpha=0.4)
    ax[1].legend(loc='upper left', labels=['Spread', 'sell_th', 'buy_th', 'sell_stop', 'buy_stop'], prop={'size':8.5})

'''
This function is similar to the one in part a.
It will compute the desired percentile volatility
and it will also selected the top certain amount from the contract
'''
def compute_aggregateLastVol_p2(rawData,ContractID,percentile_num,best_num):
    
    
    temp=rawData[['ContractId','LastVol']]
    
    #temp=temp[temp['ContractId'].isin(ContractID)]
    
    result=temp.groupby('ContractId').agg(sum)
    
    result_sort=result.sort_values(by='LastVol',ascending=False)
    
    
    percentile_vol=np.percentile(result_sort['LastVol'][result_sort['LastVol']>0],percentile_num)
    
    selected_contract=result_sort[result_sort.index.isin(ContractID)]
    top30=selected_contract.iloc[:best_num]
    
    return percentile_vol, top30




#---------------------------------------------------Backtesting function---------------------------------------------------

#To genereate set of parameter combination
def model_parameter(lookback,open_c,stop_c,close_c):

    # create configs
    configs = list()
    for i in lookback:
        for j in open_c:
            for k in stop_c:
                for l in close_c:
                    cfg = [i, j, k, l]
                    configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


#Loop over all possible parameter combination
def grid_search(all_price_data,ContractList,parameter_list,tick1,tick2,tradingIntegerAmount):
    
    
    
    ContractList=ContractList.set_index('ContractId')
    
    
    all_result = [back_testing_fullsample(all_price_data,ContractList,parameter,tick1,tick2,tradingIntegerAmount) for parameter in parameter_list]
    
    final_result=pd.DataFrame([])
    for k in range(len(all_result)):
        
        final_result=pd.concat([final_result,all_result[k]],ignore_index=True)
    
    return final_result


#Backtesting algorithm. It will simiulate the trading.
def back_testing_fullsample(all_price_data,ContractList,parameter,tick1,tick2,trading_integer_amount):
    
    
    Csize_tickx=ContractList.loc[(tick1)].values
    Csize_ticky=ContractList.loc[(tick2)].values
    
    price_1=all_price_data[tick1][0]
    price_2=all_price_data[tick2][0]
    
    lookback,open_c,stop_c,close_c=parameter
    
    
    ##We want Y to have greater value against X.
    ## Theoratically, there is not difference between X against Y or Y against X in pairs trading.
    #But in practice, using Y against X (Y has larger value),
    #will make the trading coefficient rounding more easily.
    #Because in practice, we can not trade a fraction of contract.
    
    if(price_1>=price_2):
        backtest_data=all_price_data[[tick1,tick2]].copy()
    else:
        backtest_data=all_price_data[[tick2,tick1]].copy()
    
    
    backtest_data=pd.DataFrame(backtest_data)
    backtest_data.columns=['price_y','price_x']
    
    # We will compute the contract notional amount for both contract,
    # Therefore, in practical trading, it will me more realistic.
    
    backtest_data['price_y']=backtest_data['price_y']*Csize_ticky
    backtest_data['price_x']=backtest_data['price_x']*Csize_tickx
    
    backtest_data['ln_x']=np.log(backtest_data['price_x'])
    backtest_data['ln_y']=np.log(backtest_data['price_y'])
    
    n=lookback #lookback  #8000
    open_const=open_c
    stop_const=open_c+stop_c
    close_const=open_c-close_c
    #profit 0.12
    fee=0
    
    backtest_data['residual']=0
    backtest_data['std']=0
    backtest_data['position_x']=0
    backtest_data['position_y']=0
    backtest_data['profit_x']=0
    backtest_data['profit_y']=0
    backtest_data['coef']=0

    
    #len(backtest_data)
    for i in range(n+1,len(backtest_data)):
        
        X=np.array(backtest_data['ln_x'][i-n:i-1]).reshape(-1,1)
        Y=np.array(backtest_data['ln_y'][i-n:i-1])
        
        model=LinearRegression().fit(X,Y)
        
        coef=model.coef_[0]
        
        #backtest_data['coef'].iloc[i]=coef
        
        residual=Y-model.predict(X)
        std=np.std(residual)
        
        next_X=np.array(backtest_data['ln_x'][i]).reshape(-1,1)
        next_Y=np.array(backtest_data['ln_y'][i])
        Y_hat=model.predict(next_X)
        next_residual=next_Y-Y_hat
        
        backtest_data['residual'].iloc[i]=next_residual
        backtest_data['std'].iloc[i]=std
        
        if (backtest_data['position_y'].iloc[i-1]>=0 and backtest_data['residual'].iloc[i]>= open_const*backtest_data['std'].iloc[i] and
                backtest_data['residual'].iloc[i]<=stop_const*backtest_data['std'].iloc[i]):
            #Open Position when residual is above top_close and below top_stop
            
            backtest_data['position_x'].iloc[i]=abs(coef)
            backtest_data['position_y'].iloc[i]=-1
        
        elif(backtest_data['position_y'].iloc[i-1]<=0 and backtest_data['residual'].iloc[i]<= -open_const*backtest_data['std'].iloc[i] and
                backtest_data['residual'].iloc[i]>=-stop_const*backtest_data['std'].iloc[i]):
            #Open Position when residual is below low_close and above low_stop
            
            backtest_data['position_x'].iloc[i]=-abs(coef)
            backtest_data['position_y'].iloc[i]=1
            
        elif(backtest_data['position_y'].iloc[i-1]>0 and (backtest_data['residual'].iloc[i]<-stop_const*backtest_data['std'].iloc[i] or
                backtest_data['residual'].iloc[i]>-close_const*backtest_data['std'].iloc[i])):
            
            #Close Long Position when residual is above low_close or below low_stop
            backtest_data['position_x'].iloc[i]=0
            backtest_data['position_y'].iloc[i]=0
        
        elif(backtest_data['position_y'].iloc[i-1]>0 and ~(backtest_data['residual'].iloc[i]<-stop_const*backtest_data['std'].iloc[i] or
                backtest_data['residual'].iloc[i]>-close_const*backtest_data['std'].iloc[i])):
            
            #Update Position when  long Y and neither stop condition nor close position satisfied
            backtest_data['position_x'].iloc[i]=-abs(coef)
            backtest_data['position_y'].iloc[i]=1
            
        elif( backtest_data['position_y'].iloc[i-1]<0 and (backtest_data['residual'].iloc[i]>stop_const*backtest_data['std'].iloc[i] or
                backtest_data['residual'].iloc[i]<close_const*backtest_data['std'].iloc[i]) ):
            
            #Close Short Position when residual is above top_stop or below top_close
    
            backtest_data['position_x'].iloc[i]=0
            backtest_data['position_y'].iloc[i]=0
            
            
        elif( backtest_data['position_y'].iloc[i-1]<0 and ~(backtest_data['residual'].iloc[i]>stop_const*backtest_data['std'].iloc[i] or
                backtest_data['residual'].iloc[i]<close_const*backtest_data['std'].iloc[i]) ):
        
            backtest_data['position_x'].iloc[i]=abs(coef)
            backtest_data['position_y'].iloc[i]=-1
        
        if(trading_integer_amount):
            backtest_data['position_y'].iloc[i]=backtest_data['position_y'].iloc[i]*10
            backtest_data['position_x'].iloc[i]=round(backtest_data['position_x'].iloc[i]*10)
            #backtest_data['position_x']=backtest_data['position_x'].round()
            backtest_data['profit_x'].iloc[i]=backtest_data['position_x'].iloc[i-1]*(backtest_data['price_x'].iloc[i]-backtest_data['price_x'].iloc[i-1])-abs(backtest_data['position_x'].iloc[i-1]-backtest_data['position_x'].iloc[i-2])*fee  #profit_x
            backtest_data['profit_y'].iloc[i]=backtest_data['position_y'].iloc[i-1]*(backtest_data['price_y'].iloc[i]-backtest_data['price_y'].iloc[i-1])-abs(backtest_data['position_y'].iloc[i-1]-backtest_data['position_y'].iloc[i-2])*fee  #profit_x
            
        else:
            backtest_data['profit_x'].iloc[i]=backtest_data['position_x'].iloc[i-1]*(backtest_data['price_x'].iloc[i]/backtest_data['price_x'].iloc[i-1]-1)-abs(backtest_data['position_x'].iloc[i-1]-backtest_data['position_x'].iloc[i-2])*fee  #profit_x
            backtest_data['profit_y'].iloc[i]=backtest_data['position_y'].iloc[i-1]*(backtest_data['price_y'].iloc[i]/backtest_data['price_y'].iloc[i-1]-1)-abs(backtest_data['position_y'].iloc[i-1]-backtest_data['position_y'].iloc[i-2])*fee  #profit_x
            
        
    backtest_data['total_profit']=backtest_data['profit_x']+backtest_data['profit_y']
    backtest_data['cum_profit']=np.cumsum(backtest_data['total_profit'])
    
    result=pd.DataFrame([backtest_data['cum_profit'][-1]], columns=['return'])
    result['open']=open_const
    result['close']=close_const

    result['stop']=stop_const
    result['lookback']=n

    result['tick_1']=tick1
    result['tick_2']=tick2
    
    num_x_average=np.mean(backtest_data['position_x'][backtest_data['position_x']>0])
    num_x_max=np.max(backtest_data['position_x'][backtest_data['position_x']>0])

    result['average_amount_required']=np.mean(backtest_data['price_x'])*num_x_average+10*np.mean(backtest_data['price_y'])
    result['max_amount_required']=np.mean(backtest_data['price_x'])*num_x_max+10*np.mean(backtest_data['price_y'])
    
    
    xxx=backtest_data['cum_profit'].copy()
    plt.plot(xxx.reset_index(drop=True))
    plt.title('Cumulative Return vs Time')
    return result




#------------------------------------------Main-------------------------------------------------------------------------------------

#Reading out ContractList.csv
contactlist_path='/Users/FedericoYe/Desktop/Project/MatlabQuestion/ContractList.csv'
ContractList=pd.read_csv(contactlist_path)

ContractID=ContractList['ContractId']

aPath='/Users/FedericoYe/Desktop/Project/MatlabQuestion/Functions/'
aFilePrefix='MD'
all_date=[20140519,20140520,20140521,20140522,20140523]

'''

all_new_data contains data within some time interval and 
pick up contracts in the ContractID list.

'''
all_new_data=[]
for date in all_date:
    
    all_new_data.append(filter_data(aPath,aFilePrefix,date,ContractID))

'''

all_price_data contains only the Mid price data of 38 contracts.

all_full_data contains Mid,BidVol,AskVol of 38 contracts
'''
#fullsampledata=pd.DataFrame()
all_price_data_last=pd.DataFrame()
all_price_data_mid=pd.DataFrame()

all_full_data=pd.DataFrame()
for i in range(len(all_date)):
    
    price_data_last,price_data_mid,all_data=restructure_all_price_data(all_new_data,all_date[i],i,ContractID)
    all_price_data_last=pd.concat([all_price_data_last,price_data_last])
    all_price_data_mid=pd.concat([all_price_data_mid,price_data_mid])
    
    all_full_data=pd.concat([all_full_data,all_data])


# Reorganize data to dictionary. Each key corresponds to the data of that future contract
data_by_contract={}
for i in range(len(ContractID)):
    
    data_by_contract[(ContractID[i])]=all_full_data[all_full_data['ContractId']==ContractID[i]]




#------------------------------------Examining Correlation and Cointegration to find pairs--------------------------------
first_day_data=all_price_data_last[:int(len(all_price_data_last)/len(all_date))]

#corr_matrix_s = first_day_data.corr(method='spearman').abs()
#
##the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
#sol_s = (corr_matrix_s.where(np.triu(np.ones(corr_matrix_s.shape), k=1).astype(np.bool))
#                 .stack()
#                 .sort_values(ascending=False))
#

corr_matrix_p = first_day_data.corr(method='pearson').abs()
#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
corr_pairs_p = (corr_matrix_p.where(np.triu(np.ones(corr_matrix_p.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
corr_pairs_p=pd.DataFrame(corr_pairs_p)
corr_pairs_p.columns=['correlation']


correlation_threshold=95 # This threshold means we will pick top 5% of highest correlation from all pairs
selected_pairs_corr_p=corr_pairs_p[corr_pairs_p>np.percentile(corr_pairs_p,correlation_threshold)].dropna()

#Cointegration tests and getting p-value
#The lower the p-value, the more significant will the stationary results.
all_pvalue=pd.Series(index=selected_pairs_corr_p.index)
for i in range(len(selected_pairs_corr_p)):
    first=selected_pairs_corr_p.index[i][0]
    second=selected_pairs_corr_p.index[i][1]
    _, p_value, _ = coint(first_day_data[first], first_day_data[second])
    all_pvalue[(first,second)]=p_value

selected_pairs_corr_p['pvalue']=all_pvalue
selected_pairs_corr_p=selected_pairs_corr_p.sort_values(by='pvalue', ascending=True)


#------------------------------------Sanity Check for our choice from Correlation and Cointegration--------------------------------

## This is a pair from our selection
one=selected_pairs_corr_p.index[0][0]
two=selected_pairs_corr_p.index[0][1]


idx = range(len(all_price_data_last))
temp=all_price_data_last.copy()
temp=temp.reset_index(drop=True)
plot_spread(temp, one, two, idx, 5, 10)
'''
There are two plots, the top one is the return movement of these two securities,
the second one is the ratio of two security. In this way, we can see if we can see any arbitrage opportunity.

'''
#------------------------------------Liquidity Concern----------------------------------------------

'''
This part is to check if our selection of pairs containing illiquid contract.
Among them, our selected contract pass this test.

We first compute the 70% percentile vol for those actively traded contract.
And then, we select top 30, filtering out the least couple contracts out.
'''

aDate=all_date[0]
rawData=loadRawData(aPath,aFilePrefix,aDate)


percentile_num=70
best_num=30
percentile_vol,top30=compute_aggregateLastVol_p2(rawData,ContractID,percentile_num,best_num)

print('percentile_vol is ',percentile_vol)
print('Top 30 frequently traded contract is ',list(top30.index))


#------------------------------------Parameter Backtesting----------------------------------------------


'''
Caution!!!!

stop_c/close_c here represents how many standard diviation above/below open to stop.
if open_c is 1.2, stop_c is 2,  the stopping condition 3.2.
if open_c is 1.2, close_c is 1,  the close condition 0.2.

close_c is used to get 
'''

# Model parameter
lookback=[20]#400,600,800,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000] # LSTM neruons number.
open_c=[1.2]
stop_c=[2]
close_c=[0.5]


tradingIntegerAmount=False

#lookback=[50,150,100,150,200,400,600,800,1000,2000,4000,6000,8000] # LSTM neruons number.
#open_c=[x for x in np.arange(1.1,2.6,0.3)]
#stop_c=[x for x in np.arange(0.5,1,0.2)]
#close_c=[x for x in np.arange(0.5,1,0.2)]
parameter_list=model_parameter(lookback,open_c,stop_c,close_c)


num_pairs_trade=3
for i in range(num_pairs_trade):
    
    print('pairs ',int(i+1), ':')
    print(selected_pairs_corr_p.index[i])
    
tick1=selected_pairs_corr_p.index[0][0]
tick2=selected_pairs_corr_p.index[0][1]
#tick1=141409
#tick2=681409


#Reading out ContractList.csv
ContractList=pd.read_csv(contactlist_path)

plt.figure()
final=grid_search(all_price_data_mid,ContractList,parameter_list,tick1,tick2,tradingIntegerAmount)



