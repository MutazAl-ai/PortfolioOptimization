
# coding: utf-8

# In[53]:

import numpy as np
import pandas as pd
import string as string
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import itertools
import time
import gc
import datetime









def Filter_returns(comPort, symbols,returns):        
        for symbol in symbols:
            if symbol not in comPort:
                returns= returns.T.drop(symbol, axis=0).T
        
        return returns





def symbol_to_path(symbol, base_dir="c:/data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
            
        
    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0

    

    
    return daily_returns.fillna(value=0)


def symbol_to_path_dividends(symbol, base_dir="c:/data/dividends"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data_dividends(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')
    
    for symbol in symbols:
        if symbol == 'SPY':
                df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
                df_temp = df_temp.rename(columns={'Close': symbol})
                df = df.join(df_temp)
        else:    
            df_temp = pd.read_csv(symbol_to_path_dividends(symbol), index_col='Date',
                    parse_dates=True, usecols=['Date', 'Dividends'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Dividends': symbol})
            df = df.join(df_temp)

        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
        
    return df



# In[72]:

dates = pd.date_range('2013-1-1', '2016-1-1')  # 60% train and 40% cross-validate
symbols = [  'SDS',  'FLGE','DVYL', 'HYEM1', 'KBWY', 'PGX', 'PSI']
dividendsSymbols = [ 'DVYL', 'KBWY','HYEM1',  'PGX']




df = get_data(symbols, dates)
dv = get_data_dividends(dividendsSymbols, dates)

dv = dv.fillna(0)
dv = dv.cumsum()

dfadd=df + dv

SPYreturnPeriod = df['SPY'] [-1] / df['SPY'] [0] - 1
print( SPYreturnPeriod)

for symbol in symbols:
    if symbol not in dividendsSymbols or symbol=='SPY':
        dfadd[symbol] = df[symbol]

df= dfadd
# Compute daily returns
returns = compute_daily_returns(df)

plot_data(returns, title="Daily returns", ylabel="Daily returns")



# In[73]:

cleanedCovar = df
       
cleanedReturns = compute_daily_returns(cleanedCovar)

covar = np.cov(cleanedReturns.T)
corr = np.corrcoef(cleanedReturns.T)

covar= np.cov(covar)
corrDf = pd.DataFrame(corr)

print(corrDf.mean().mean())

mean_return = returns.mean()
return_stdev = returns.std()
corrDf = pd.DataFrame(corr, index=mean_return.index, columns=mean_return.index)

print(corrDf)

#print(covar)


# In[74]:


 
annualised_return = mean_return * 252
annualised_stdev = return_stdev * np.sqrt(252)

print ("annualized return \n{}".format(annualised_return))
print ("annualized Standard deviation \n{}".format(annualised_stdev))
print(annualised_return / annualised_stdev)


# In[75]:

def Compute_Comulative_returns(principal, dfReturnsFiltered, optimalW):
    counter=0
    dfReturnsFiltered = dfReturnsFiltered+1
    
    for index, row in dfReturnsFiltered.iterrows():
        columnIndex= 0
       
        for c in dfReturnsFiltered: 
            if counter==0:
            
                val = optimalW[columnIndex]*principal
                dfReturnsFiltered.set_value(index,c,val )

                columnIndex+=1
            else:

                previousValue = dfReturnsFiltered.iloc[counter-1][columnIndex]
                dfReturnsFiltered.loc[index, c] =  dfReturnsFiltered.loc[index, c] * previousValue
                columnIndex+=1
             

        counter+=1
    
    return dfReturnsFiltered





def compute_portfolio_weights(local_comReturnsDf):
    counter=0

    returnDf = pd.DataFrame(local_comReturnsDf)
    returnDf["NAV"] = returnDf.sum(axis=1) 
    
    for index, row in returnDf.iterrows():
        columnIndex= 0
        
        for c in returnDf:          
                val = returnDf.loc[index, c] / row['NAV']
                returnDf.set_value(index,c,val )
             
        counter+=1
    
    returnDf = returnDf.drop('NAV', 1)
 
    return returnDf



    

def Compute_Comulative_returns_single(principal, daily_returns_by_W):
    comulativeReturns= [None]*len(daily_returns_by_W)
    counter = 0
    
    for r in daily_returns_by_W:
        if counter>0:
             comulativeReturns[counter] = comulativeReturns[counter - 1] * (1+r)

        else:
            comulativeReturns[counter] = principal * (1+r)
            
        counter+=1
    
    return np.array(comulativeReturns)



def return_random_weights(hedge, CombSymbols):
    k = np.random.rand(len(CombSymbols)-1)
    sumK= sum(k) 
    randW = (k / sumK) *(1-hedge)
    return randW 

def return_random_weights_no_hedge( len_comb_symbols):
    k = np.random.rand(len_comb_symbols)
    sumK= sum(k) 
    randW = (k / sumK)
    return randW 

#improve the random function to create weights with combinations rather than completely random
def Create_weights(hedge, CombSymbols):
        returnsWeights=[None]*NoOfSims

        for i in range(0, NoOfSims):
            randW = return_random_weights(hedge, CombSymbols)
            tempRandW = randW.tolist()
            tempRandW.insert(SDSLocation, hedge)
            randW = np.array(tempRandW)

    
            returnsWeights[i]=randW
        return returnsWeights
    
def Create_weights_no_hedge( len_comb_symbols):
        returnsWeights=[None]*NoOfSims

        for i in range(0, NoOfSims):
            randW = return_random_weights_no_hedge(len_comb_symbols)
    
            returnsWeights[i]=randW
        return returnsWeights
    
    
def Compute_Daily_Return_Comulative(Comulative_returns_df):
        Comulative_returns_sum= Compute_Comulative_returns(principal, filteredReturnsDf, weights).sum(axis=1)
        Comulative_daily_returns = (Comulative_returns_sum / Comulative_returns_sum.shift(1)) - 1
        
        return Comulative_daily_returns
        
        
def Save_DataFrame(name, superArray):
    latestFileName = "d:/output/{}{:%Y-%m-%d %H%M%S}.csv".format(name,datetime.datetime.now())

    dataFrame = pd.DataFrame(superArray)

    dataFrame.to_pickle(latestFileName)
    
    return latestFileName
    
def Read_saved_DataFrame(full_name):
    ss = pd.read_pickle(full_name) 
    
    return ss

principal = 100000


# Ensemble of Algorithima
# 
# 1. Q-learning to decide weights using optimal months history
# 
#     - What is the weight combination that gives the best return for the month after?
# 
#     - How Many months we had to go back to get the weights?
# 
#     - What is the pattern and distribution of months needed to give best weights?
# 
#     - Can we predict the optimal months for getting optimal weights?
# 
#     - How much is the error in prediction costing us?
# 
# 
# 2. Momentum indicator for overbought and oversold
# 
# 

# In[76]:

def Compute_Comulative_returns_rebalanced(principal, local_dfReturnsFiltered, optimalW):
    local_dfReturnsFiltered = local_dfReturnsFiltered+1
    
    megaArray = local_dfReturnsFiltered.as_matrix()
    
    rows= megaArray.shape[0]
    cols= megaArray.shape[1]
    
    for rowIndex in range(0,rows):  
        for columnIndex in range(0,cols): 
            
            if rowIndex==0:
                val = optimalW[columnIndex]*principal
                megaArray[rowIndex,columnIndex]= val 

            else:

                previousValue = megaArray[rowIndex-1][columnIndex]
                megaArray[rowIndex,columnIndex] =  megaArray[rowIndex,columnIndex] * previousValue
                
        #rebalance
        columnIndex= 0
        for columnIndex in range(0,cols): 
            if rowIndex>0:
                #sliding window balancing is a good one
                #find the period that gives best stock selection and weight return for the period after
                #for example 3 months rebalancing based on last month, or 6 months history to exlplain 3 or 6 monthis history for balancing 
                #if we pick say 3 months, a months after the new month based will be incorporated
                #this is where the magic should happen, Q-learning and statistical arbitrage, Std deviation adn correlation heat-up
                # what is the best weight for the portfolio? which period has defined it?
                #1- we will try 3 months, take weights, test, recod, then 4, then 5, etc...
                #2- then slide to next month, also try 3, 4 to say 24,
                #3- that the distribution of the months needed, for example a mode of 6 months gave the best weights 
                #4- look for patters, when do the number of months to give best weights for optimal returns change? why? could we forecast that change based on information?
                NAV = np.sum(megaArray[rowIndex])
                current_weight = megaArray[rowIndex,columnIndex] / NAV
                adjustment = (optimalW[columnIndex] - current_weight) * NAV #here should be a q-learning function that finds the optimal adjustment
                megaArray[rowIndex,columnIndex] = megaArray[rowIndex,columnIndex] + adjustment

    
    df = pd.DataFrame(megaArray, columns =local_dfReturnsFiltered.columns, index =local_dfReturnsFiltered.index )
    
    
    return df


# In[77]:

import time
import gc


principal = 100000
NoOfSims = 1000 # ideally 10,000 to 15,000 sims
HedgeCounter = 0
fleshedPorts = []
sqrt =  np.sqrt(252)

projection_days = 21
sliding_days_step = 21
history_days_step = 21



minimum_history = 2 * history_days_step
maxumum_history = 12 * history_days_step + 1

max_slide_days = 12 * sliding_days_step + maxumum_history + 1  # should be 12 months full run, except here it is sliding 1 months only

comPorts=[]

recordCounter = 0
loopCounter = 0 
timer_first_time = True

hedge_Max= 0.2+0.01
hedge_step = 0.05

for c in range(5, 7):
    for L in range(0, len(symbols)+1):
            for subset in itertools.combinations(symbols,c):
                if 'SDS' in subset and 'SPY' not in subset:
                #if  'SPY' not in subset  :
                    comPorts.append(subset) 

Length_counter=0
                    
for start_day_adjustment in range(maxumum_history,max_slide_days,sliding_days_step): 
    for days_history in range(minimum_history, maxumum_history, history_days_step): # 50 days to a year
        for h in np.arange(0.05,hedge_Max, hedge_step):
                           Length_counter+=1;
            
            
iterations = len(comPorts) * NoOfSims * Length_counter 
print(iterations)  
#'f8, f8,f8,O,O, f8, f8,i, f8,f8,M'
superArray = np.empty([round(iterations),10],dtype=object)

        
#superArray=[]
    




#this only tests one month, we need a sliding windows to get to month predictability patters
#this should answer many questions like: what is the best period to give weights to maximize next month return for that month
#sliding will give each month best period to give weights to maximize next month?
#the patters of months needed will give insights, the average or mode could maximize our rebalancing
#also, the months to give best weights for maximum return can be modelled instead of using mode or average? modeled based on what?
startTime = time.time()
s = time.time()
for start_day_adjustment in range(maxumum_history,max_slide_days,sliding_days_step): 
    
    for days_history in range(minimum_history, maxumum_history, history_days_step): 
        
        SPYreturns = np.array(returns['SPY'][start_day_adjustment-days_history:start_day_adjustment])
        SPY_var = np.var(SPYreturns)

        for comPort in comPorts:
            
            filtered_returns_total_hist = Filter_returns(comPort,symbols,returns)
            filtered_returns = filtered_returns_total_hist[start_day_adjustment-days_history:start_day_adjustment]
            
            
            future_returns = filtered_returns_total_hist[start_day_adjustment:start_day_adjustment+projection_days]
            future_returns_arr = np.array(future_returns)
            
            projected_first_day = future_returns.index[0]

            Treturns = np.nan_to_num(filtered_returns)

            comPort = list(filtered_returns.columns)
            SDSLocation = comPort.index("SDS")
            for h in np.arange(0.05,hedge_Max, hedge_step):   
                
            #weightsDF=pd.DataFrame(Create_weights_no_hedge(len(comPort))).T
                weightsDF=pd.DataFrame(Create_weights(h,comPort)).T


                for w in weightsDF:
                    
                    loopCounter+=1
                    weights = weightsDF[w]

                    ###################
                    # moving the two outer loops to find optimal weights inside can give us better indicator, worth trying
                    # we can use association to get to the period by identifying the conditions that are closest to to the current day from history generated is this loop
                    # we can use weights, return for stocks indvidually, market conditions, correlation, std dev for association to find the period that worked best to get optimal wrights
                    # but first we use average period that brought highest return, apply check if we get better returns for the 12 months after
                    # then we do matching as in the third point to pick which is the best period and retest
                    #ideally the average days use with highest return will enahnce rebalancing weights and returns, association will enahance it further
                    # in total this in effect is telling us what to sell or buy
                    # we can also use a mid-point or alpha smoothing between new rebalanced weights in the portfolio and old weights, that may give better returns
                    #####################
            
                    

                    #if(meanR_projection  > 0.12):
                    daily_returns_by_W = np.dot(Treturns, weights)
                    meanR_hist=daily_returns_by_W.mean() * 252
                        
                    if(meanR_hist > 0.12):
                        meanR_projection = np.dot(future_returns_arr, weights).mean() *252 
                        meanR_projection = np.dot(future_returns, weights).mean() * 252

                        beta = (np.cov(daily_returns_by_W, SPYreturns)[0,1]) / SPY_var


                        #if beta>-1 and beta <1:
                        stdDevI =  daily_returns_by_W.std() * sqrt
                        #use array
                        #superArray= [meanR,stdDevI,beta, weights, comPort,mean balanced return, balanced_beta, days, start_date]
                        superArray[recordCounter][0]= meanR_hist# 5 is for mean balanced return, 6 is for balanced_beta, 7 is reserved 
                        superArray[recordCounter][1]=stdDevI
                        superArray[recordCounter][2]=beta
                        superArray[recordCounter][3]=weights
                        superArray[recordCounter][4]=comPort
                        superArray[recordCounter][5]=0
                        superArray[recordCounter][6]=0
                        superArray[recordCounter][7]=days_history
                        superArray[recordCounter][8]=meanR_projection
                        superArray[recordCounter][9]=projected_first_day

                            #superArray.append([meanR_hist, stdDevI, beta,weights,comPort,0,0,days_history, meanR_projection,projected_first_day ])

                        recordCounter+=1

            e=time.time()
            if timer_first_time and (e-s)> 30:            
                   seconds = (e-s) 
                   print("{} in {} seconds, Takes about {} mins".format(loopCounter, round(seconds) , round((seconds * iterations/60)/loopCounter) ))
                   timer_first_time = False
             
            else:
                e=time.time()
                if (e-s)> 30:
                    seconds = (e-s) 
                    print("Processed%:{}, Processed:{} in {}, added:{} Time Elapsed: {} mins".format(round((loopCounter/iterations)*100),round(loopCounter),round(seconds), recordCounter, round((e-startTime)/60)))
                    s=time.time()
                    
                    
    
superArray.resize((recordCounter, 10))

print("done")


# In[9]:

import math



name = "OptiP"
column_names =[ 'meanR_hist', 'stdDevI', 'beta','weights','comPort','meanR_balanced','beta_balanced','days_history', 'meanR_projection','projected_first_day' ]

tempDf = pd.DataFrame(superArray, columns=column_names)


tempDf[['meanR_hist','stdDevI','beta','meanR_balanced','beta_balanced','days_history', 'meanR_projection']]= tempDf[['meanR_hist','stdDevI','beta','meanR_balanced','beta_balanced','days_history', 'meanR_projection']].apply(pd.to_numeric)



#latestFileName=Save_DataFrame(name,superArray)

latestFileName = "d:/output/{}{:%Y-%m-%d %H%M%S}.csv".format(name,datetime.datetime.now())

#dataFrame = pd.DataFrame(superArray)
#s=time.time()
#print("started writing file")
#tempDf.to_csv(latestFileName, index = False, chunksize=500000)


#print("done", s-time.time())


# In[17]:

#latestFileName = "d:/output/OptiP2017-07-17 105917.csv" 




#ss = pd.read_pickle(latestFileName) 
#print("done")

#print(ss.head())


# In[15]:

#del ss['error']
#ss = ss[1:]
ss= tempDf
smaller = ss[['days_history', 'meanR_projection','projected_first_day']]

#print(smaller.groupby(['projected_first_day','days_history']).mean())

smaller.groupby(['days_history']).mean()


#plt.show()


# In[26]:

optimal =[]
count_short_list = 0
stdDev_threshold = 0.1
beta_threshold= 0.2
max_weight =0.3
timer_first_time = True


superArray= ss[(ss.days_history==210)].as_matrix()
len(superArray)
processed_count=0
add_count=0

for element in superArray:
    if element[1]<stdDev_threshold and element[2]<beta_threshold:
        weights = element[3]
        filtered = [i for i in weights if i < max_weight] 
        if len(filtered)==(len(weights)):
            count_short_list+=1

print(count_short_list)      

short_list_array = count_short_list*[None]

s=time.time()
for element in superArray:
    
    
    if element[1]<stdDev_threshold and element[2]<beta_threshold:
        
        weights = element[3]
        comPort= element[4]
        days_history=element[7]
        filtered = [i for i in weights if i < max_weight]  
      
        if len(filtered)==(len(weights)):
            SPYreturns = np.array(returns['SPY'].head(n=days_history))
            
            
            filtered_returns_total_hist = Filter_returns(comPort,symbols,returns)
            filtered_returns = filtered_returns_total_hist.head(n=days_history)
            Comulative_returns_rebalanced = Compute_Comulative_returns_rebalanced(principal, filtered_returns, weights)
            Comulative_returns_rebalanced_sum = Comulative_returns_rebalanced.sum(axis=1)
            Comulative_returns_rebalanced_daily = (Comulative_returns_rebalanced_sum / Comulative_returns_rebalanced_sum.shift(1)) - 1
            Comulative_returns_rebalanced_daily[0] = 0
            
            beta_balanced = (np.cov(Comulative_returns_rebalanced_daily, SPYreturns)[0,1]) / SPY_var
        
            meanR_balanced = Comulative_returns_rebalanced_daily.mean() * 252

            #if meanR_balanced > 0.00 and beta_balanced <0.7 and beta_balanced>-0.7:
            element[5] = meanR_balanced
            element[6] = beta_balanced
            optimal.append(element)
            add_count+=1
         
            if timer_first_time:
                   e=time.time()
                   seconds = (e-s) * count_short_list
                   print("Takes about {} mins".format(round(seconds/60),2))
                   timer_first_time = False
            else:
                e=time.time()
                if (e-s)> 15:
                    print("Processed%:{}, Added:{}".format((processed_count/count_short_list)*100,add_count))
                    s=time.time()
                    
            processed_count+=1
        
          
print("done")


# In[29]:

v = np.array(optimal,  dtype=object)

#maxExcess = np.array(v[np.where(v[:,3] ==v[:,3].min())])

q = pd.DataFrame(optimal)

maxReturn =v[np.where(q[5] ==q[5].max())]

optimalW = maxReturn[:,3][0]
optimalNames=maxReturn[:,4][0]



frames = [optimalNames, optimalW]
f = np.column_stack(frames)

print("return {} Risk {} Beta {} Adjusted Return{} Adjusted Beta {}".format(maxReturn[:,0], maxReturn[:,1], maxReturn[:,2], maxReturn[:,5], maxReturn[:,6]))


print(f)





# In[34]:


SPYreturns =  np.array(returns['SPY'])


filteredReturns = Filter_returns(optimalNames,symbols,returns )

comReturnsDf = Compute_Comulative_returns(principal, filteredReturns, optimalW)


comulativeReturns1= comReturnsDf.sum(axis=1)

portfolio_weightsdf=  compute_portfolio_weights(comReturnsDf)
    
    
plot_data(portfolio_weightsdf)


balanced_returns_df=Compute_Comulative_returns_rebalanced(principal, filteredReturns, optimalW)



SPYcomulative= Compute_Comulative_returns_single(principal, SPYreturns) 




comR=pd.DataFrame(comulativeReturns1)
SpyR=pd.DataFrame(SPYcomulative)

plot_data(balanced_returns_df, title="Balanced")

balancedReturnsDf = balanced_returns_df.sum(axis=1)

frames2 = [comR, SpyR,balancedReturnsDf ]

ff = np.column_stack(frames2)



#print(principal)

plot_data(pd.DataFrame(ff, columns =["Natural", "S&P", "Rebalanced"]), title="Portoflios against S&P",xlabel="Date", ylabel="Portfolio Value")

balanced_weights = compute_portfolio_weights(balanced_returns_df)
plot_data(balanced_weights, title="Balanced Weights")
    


# In[35]:

Omptimal_returns_w = (comulativeReturns1 / comulativeReturns1.shift(1)) - 1
Omptimal_returns_w =  Omptimal_returns_w.fillna(value=0)


plt.hist(Omptimal_returns_w, bins=20)
plt.show()

print("mean annual return (not accumalitive) {}".format(np.nan_to_num(Omptimal_returns_w).mean()* 252))


Omptimal_returns_w.describe()



# In[36]:

balanced_daily_returns = (balancedReturnsDf / balancedReturnsDf.shift(1)) - 1
balanced_daily_returns =  balanced_daily_returns.fillna(value=0)

plt.hist(balanced_daily_returns, bins=20)
plt.show()

print("mean annual return (not accumalitive) {}".format(np.nan_to_num(balanced_daily_returns).mean()* 252))
balanced_daily_returns.describe()


# In[37]:


model = sm.OLS(balanced_daily_returns, returns['SPY']).fit()


print(model.summary())



# In[38]:

dates2 = pd.date_range('2014-6-30', '2017-06-24')  # one month only
symbols2 = symbols
dividendsSymbols2 = dividendsSymbols

df2 = get_data(symbols2, dates2)



dv2 = get_data_dividends(dividendsSymbols2, dates2)



dv2 = dv2.fillna(0)
dv2 = dv2.cumsum()

dfadd2=dv2 + df2

for symbol2 in symbols2:
    if symbol2 not in dividendsSymbols2 or symbol2=='SPY':
        dfadd2[symbol2] = df2[symbol2]




df2= dfadd2

returns2 = compute_daily_returns(df2)

returns_filtered2= Filter_returns(optimalNames,symbols, returns2)






# In[45]:


SPYreturns2 =  np.array(returns2['SPY'])
SPYcomulative2 = []
comulativeReturns2=[]
counter = 0

comulative_returns_df2 =   Compute_Comulative_returns(principal, returns_filtered2, optimalW)
    
comulativeReturns2= comulative_returns_df2.sum(axis=1)
SPYcomulative2= Compute_Comulative_returns_single(principal, SPYreturns2) 

balanced_returns_df2=Compute_Comulative_returns_rebalanced(principal, returns_filtered2, optimalW)




balanced_returns_df2_sum= balanced_returns_df2.sum(axis=1)


frames22 = [comulativeReturns2, SPYcomulative2, balanced_returns_df2.sum(axis=1) ]

ff2 = np.column_stack(frames22)



plot_data(Compute_Comulative_returns(principal, returns_filtered2, optimalW))


print(comulativeReturns2[-1])

plot_data(pd.DataFrame(ff2, columns =["Natural", "S&P", "Rebalanced"]))
print("Std in USD {}".format(pd.DataFrame(comulativeReturns2).std()))
print("Std % {}".format(pd.DataFrame(comulativeReturns2).std() / pd.DataFrame(comulativeReturns2).mean()))


print("Pre Std in USD {}".format(pd.DataFrame(comulativeReturns1).std()))
print("Pre Std % {}".format(pd.DataFrame(comulativeReturns1).std() / pd.DataFrame(comulativeReturns1).mean()))


# In[40]:

daily_returns_by_W2 = (comulativeReturns2 / comulativeReturns2.shift(1)) - 1
daily_returns_by_W2 =  daily_returns_by_W2.fillna(value=0)


plt.hist(daily_returns_by_W2, bins=20)
plt.show()

print("mean annual return (not accumalitive) {}".format(np.nan_to_num(daily_returns_by_W2).mean()* 252))


daily_returns_by_W2.describe()


# In[41]:

balanced_daily_returns2 = (balanced_returns_df2_sum / balanced_returns_df2_sum.shift(1)) - 1
balanced_daily_returns2 =  balanced_daily_returns2.fillna(value=0)

plt.hist(balanced_daily_returns2, bins=20)
plt.show()

print("mean annual return (not accumalitive) {}".format(np.nan_to_num(balanced_daily_returns2).mean()* 252))
balanced_daily_returns2.describe()


# In[42]:



model = sm.OLS(daily_returns_by_W2,returns2['SPY']).fit()
print(model.summary())


model = sm.OLS(balanced_daily_returns2,returns2['SPY']).fit()
print(model.summary())




# In[43]:

corr = comulative_returns_df2.rolling(window=10).corr(comulative_returns_df2, pairwise=True)[:,0].T
corr = corr.mean(axis=1)


stdsMov10 = pd.DataFrame(pd.rolling_std(comulativeReturns2, window=10)*np.sqrt(252))
stdsMov60 = pd.DataFrame(pd.rolling_std(comulativeReturns2, window=60)*np.sqrt(252))

priceMov = pd.DataFrame(pd.rolling_mean(comulativeReturns2, window=5))

plt.plot(stdsMov10, 'r')
plt.plot(stdsMov60, 'g')
plt.show()

plt.plot(corr)
plt.show()
print(np.nan_to_num(corr).mean())


# In[44]:

corrB = balanced_daily_returns.rolling(window=10).corr(balanced_daily_returns, pairwise=True)[:,0].T
corrB = corrB.mean(axis=1)



stdsMov10B = pd.DataFrame(pd.rolling_std(balanced_daily_returns, window=10)*np.sqrt(252))
stdsMov60B = pd.DataFrame(pd.rolling_std(balanced_daily_returns, window=60)*np.sqrt(252))

priceMovB = pd.DataFrame(pd.rolling_mean(balanced_daily_returns, window=5))

plt.plot(stdsMov10B, 'r')
plt.plot(stdsMov60B, 'g')
plt.show()

plt.plot(corrB)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



