# Project_Bonds
Project Title : Price forecasting of SGB and IRFC Bonds and comparing there returns.
## Introduction of the Project
The 2008-09 global financial crises and 2020-21 pandemic have shown us the volatility of the market. Many people have are finding a way to invest money to secure their future. People are trying to find a secure investment with minimum financial risks with higher returns. This is also a fact that with investment their also comes with risks. There is a saying in the world of investment “Do not put all your egg in one basket”. We need to diverse portfolio in the area of investment, so that if one investment does not give you enough yields due to fluctuations in the market rates then other will give you higher yield. Bonds are one such investment people prefer the most. The Bonds we have selected are two government bonds – SGB (Sovereign Gold Bond) and IRFC (Indian Railway Finance Corporation). The objective was to forecast the prices of SGB and IRFC bond and calculate the returns. Compare the returns and recommend the client which one to pick based on the input that is number of years to forecast.

## Technologies Used
* Python – ML model (auto_arima (for grid search to find p,q,d values), ARIMA(for forecasting values))<br>
* SQLite – Database<br>
* Flask – Front End for deployment<br>
* Python Libraries – numpy, pandas, Statsmodels, re, nsepy, matplotlib<br>
* HTML/CSS<br>

## General info
This project is simple Forecasting model. Not taxes were put into use when calculating returns. IRFC Bond is a tax free bond but SGB we need to pay taxes if we try to sell it before the maturity period is over.
Inflation rate and global pandemic situation is a rare phenonmenon and it is beyond anyone's control. It has been taken into business restriction.<br>
Data has been collected from [National Stock exchange of India](https://www1.nseindia.com/index_nse.htm)
The two bonds selected from NSE was -
* [Indian Railway Finance Corporation Limited Bond (IRFC)](https://www1.nseindia.com/live_market/dynaContent/live_watch/get_quote/GetQuote.jsp?symbol=IRFC&series=N2)
* [Sovereign Gold Bonds(SGBAUG24)](https://www1.nseindia.com/live_market/dynaContent/live_watch/get_quote/GetQuote.jsp?symbol=SGBAUG24&illiquid=0&smeFlag=0&itpFlag=0)


## Requirement file (contains libraries and their versions)
[Libraries Used](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/requirements.txt)

## Project Architecture
![alt text](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/project_arch.png)

## Explaining Project Architecture
### Live data extraction
The data collected from NSE website (historical data) and the library which is used to collect live daily data from the website is [nsepy](https://nsepy.xyz/). The data is then goes to python, two things happens in python. First, out of all the attributes, we only take "Close Price" and then the daily is then converted into monthly data. We use mean to calculate the the monthly average.
### Data storage in sqlite 
We chose SQLite because it is very easy to use and one does not need the knowledge of sql to observe the data. the database is created locally and and is being updated when the user usses the application. the user can easliy take the database and see the data in SQL viewr online available.
### Data is then used by the model
When data is then called back by the python. the python then perform differencing method to remove the trend and seasonality from the data so that our data can be stable. For successful forecasting, it is necessary to keepp the time series data to be stationary.
#### p,d,q Hyperparameters
We use auto_arima function to calculate p,d,q value. We use re(regex) to store the summary of auto_arima in string format. then use "re.findall()" funtion to collect the value of p,d,q values. The downpoint of using this auto_arima function is that it runs two times when the programes gets executed. It calculate the hyperparameter values for both SGB and IRFC data.
### ARIMA
This part is where the data is taken and then fit & predict.<br>
This is for 12 months.
![Actual Data vs Predicted Data](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/actualvspred.png)
### Model Evaluation
#### SGB
The RMSE: 93.27 Rs. & The MAPE: 0.0185
#### IRFC
The RMSE: 21.62 Rs. & The MAPE: 0.0139<br>
(Pretty Good)
### Forecasting (12 Months)
![Forecasted Data (12 Months)](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/forecast.png)
### Returns
This is the part where both SGB and IRFC foecasted data is being collected and based on that returns are calculated. If the SGB returns is higher than IRFC bonds then it will tell the customer about the amount of return for a specific time period.
### User Input
The user will be given 3 options as Input. The user will select a specific time period from a drop down list. The options are -<br>
1. 4 Months (Quaterly)<br>
2. 6 Months (Half yearly)<br>
3. 12 Months (Anually)<br>
This options are time pperiod to forecast. If the user press 6 then the output page will show "6" forecasted values with a range Upper Price, Forecasted Price, Lower Price for both the bonds side by side. Below there will be a text where the returns will be diplayed if the user decides to sell the bonds then.<br>
12 Months Forecasted Prices -
![forecasted_prices](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/forecastedprice.png)


## Python_code
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import math
import re
from datetime import date
import nsepy 
import warnings
warnings.filterwarnings("ignore")
####################################          Live data extraction              ###################################################
##Extracting data from nsepy package
da=date.today()
gold= pd.DataFrame(nsepy.get_history(symbol="SGBAUG24",series="GB", start=date(2016,9,1), end=da))
bond= pd.DataFrame(nsepy.get_history(symbol="IRFC",series="N2", start=date(2012,1,1), end=da))

#############################                 Live data  extraction end                  ###############################################

# Heatmap - to check collinearity
def heatmap(x):
    plt.figure(figsize=(16,16))
    sns.heatmap(x.corr(),annot=True,cmap='Blues',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()
heatmap(gold)
heatmap(bond)
###############################                Live data to Feature engineering            ################################################3             

##Taking close price as our univariate variable
##For gold
gold=pd.DataFrame(gold["Close"])
gold["date"]=gold.index
gold["date"]=gold['date'].astype(str)
gold[["year", "month", "day"]] = gold["date"].str.split(pat="-", expand=True)
gold['Dates'] = gold['month'].str.cat(gold['year'], sep ="-")
gold.Dates=pd.to_datetime(gold.Dates)
gold.set_index('Dates',inplace=True)
col_sgb=pd.DataFrame(gold.groupby(gold.index).Close.mean())

##For bond
bond=pd.DataFrame(bond["Close"])
bond["date"]=bond.index
bond["date"]=bond['date'].astype(str)
bond[["year", "month", "day"]] = bond["date"].str.split(pat="-", expand=True)
bond['Dates'] = bond['month'].str.cat(bond['year'], sep ="-")
bond.Dates=pd.to_datetime(bond.Dates)
bond.set_index('Dates',inplace=True)
col_bond=pd.DataFrame(bond.groupby(bond.index).Close.mean())

col_sgb.columns = ["Avg_price"]
col_bond.columns = ["Avg_price"]

col_bond.isnull().sum()
col_sgb.isnull().sum()

############################                  SQL connection with monthly data           ################################################ 
###############################                SQL database is created                  ################################################3             

# Connect to the database
from sqlalchemy import create_engine
engine_sgb = create_engine('sqlite:///gold_database.db', echo=False)
col_sgb.to_sql('SGB', con=engine_sgb,if_exists='replace')
df_sgb = pd.read_sql('select * from SGB',engine_sgb )

df_sgb.Dates=pd.to_datetime(df_sgb.Dates)
df_sgb.set_index('Dates',inplace=True)


engine_irfcb = create_engine('sqlite:///irfcb_database.db', echo=False)
col_bond.to_sql('IRFCB', con=engine_irfcb,if_exists='replace')
df_bond = pd.read_sql('select * from IRFCB',engine_irfcb)

df_bond.Dates=pd.to_datetime(df_bond.Dates)
df_bond.set_index('Dates',inplace=True)
###############################                SQL data to python                 ################################################3             



# Plotting
def plotting_bond(y):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Monthly Average')
    ax.plot(y.resample('Y').mean(),marker='o', markersize=8, linestyle='-', label='Yearly Mean Resample')
    ax.set_ylabel('Avg_price')
    ax.legend();
plotting_bond(df_sgb)
plotting_bond(df_bond)

#univariate analysis of Average Price
df_sgb.hist(bins = 50)
df_bond.hist(bins = 50)

# check Stationary and adf test
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label = "Original Price")
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std');
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation - Removed Trend and Seasonality')
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    print('Test statistic = {:.3f}'.format(adft[0]))
    print('P-value = {:.3f}'.format(adft[1]))
    print('Critical values :')
    for k, v in adft[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<adft[0] else '', 100-int(k[:-1])))
    
test_stationarity(df_sgb)
test_stationarity(df_bond)
        

# Differencing method to remove trend and seasonality
diff_sgb = df_sgb - df_sgb.shift()
diff_sgb.dropna(inplace = True)
test_stationarity(diff_sgb)

diff_bond = df_bond - df_bond.shift()
diff_bond.dropna(inplace = True)
test_stationarity(diff_bond)

#Decomposition
dec_sgb= seasonal_decompose(df_sgb)  
dec_bond = seasonal_decompose(df_bond) 
dec_sgb.plot() 
dec_bond.plot()


#------- plotting Moving avarage with df_sgb/bond and diff_sgb/bond--------
def movin_avg(inpt): 
    rcParams['figure.figsize'] = 10, 6
    moving_avg = inpt.rolling(12).mean()
    std_dev = inpt.rolling(12).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(inpt, label = "Avr_Price")
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend()
    plt.show()
movin_avg(df_sgb)
movin_avg(diff_sgb)
movin_avg(df_bond)
movin_avg(diff_bond)

#------ Finding p, d, q values from acf and pcf plot -------
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# for SGB
plot_acf(diff_sgb, lags = 12) #AR
plot_pacf(diff_sgb, lags = 12) #MA
#for irfc
plot_acf(diff_bond, lags = 12) #AR
plot_pacf(diff_bond, lags = 12) #MA
# d = 1

''' n = forecated time period (given by the user) '''
n = 4 #quaterly
n = 6 #Halfyearly
n = 12 # Yearly

''' Finding p, d, q hyperparameter from auto_arima '''
def arima_mod(df_, diff_,x): ######### change y to df_
    model_autoARIMA = auto_arima(diff_, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=12,              # 1/frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # False/No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    summary_string = str(model_autoARIMA.summary())
    param = re.findall('SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)',summary_string)
    p,d,q = int(param[0][0]) , int(param[0][1]) , int(param[0][2])
    print(p,d,q) 

    ''' ARIMA model '''
    model = ARIMA(df_, order=(p,1,q), trend = "t") 
    fitted = model.fit()
    print(fitted.summary())
    ''' Predicting the actual value '''
    line_pred = fitted.predict()
    line_pred = line_pred[1:len(line_pred)]

    plt.figure(1)
    plt.plot(df_, label = "Actual Value")
    plt.plot(line_pred, color = "red", label = "predicted Value by Model")
    plt.legend(loc = "best")
    
    ''' RMSE and Mape'''
    rmse = math.sqrt(mean_squared_error(df_[1:len(df_)],line_pred))
    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(df_[1:len(df_)], line_pred)
    print('RMSE: '+str(rmse)) 
    print('MAPE: '+str(mape)) 

    ''' Forecating '''
    forecasted_val = fitted.predict(1,len(df_)+(x-1)) 
    forecast_next= fitted.forecast(steps = x, alpha = 0.05) #95 confidence interval
    # Confidence interval of 95%
    forecast = fitted.get_forecast(x, alpha = 0.05)
    conf = forecast.conf_int(alpha=0.05)
    # storing the confidence interval in a series
    lower_series =pd.Series(conf["lower Avg_price"], index=forecasted_val.index[(len(df_) - 1):])
    upper_series =pd.Series(conf["upper Avg_price"], index=forecasted_val.index[(len(df_) - 1):])

    ''' Plot Actual Value and Forecasted Value '''
    plt.figure(2)
    plt.figsize=(20, 6)
    plt.plot(df_, label = "Actual")
    plt.plot(forecast_next, color = "red", label = "forecasted")
    plt.legend(loc = "best")
    plt.xlabel("Years")
    plt.ylabel(" Average_Price")
    plt.fill_between(lower_series.index, lower_series, upper_series,color='k', alpha=.05)
    
    Forecast_series_ = pd.concat([lower_series,forecast_next,upper_series], axis = 1)
    Forecast_series_.columns = ["Lower_value","Forecasted_value","Upper_value"]
    
  
    return Forecast_series_

store_sgb = arima_mod(df_sgb, diff_sgb, n) # Storing the Forecated,Upper and lower price of SGB
store_bond = arima_mod(df_bond,diff_bond,n) # Storing the Forecated,Upper and lower price of irfc

''' Forecasted Price, Upper Price and Lower Price plot '''
store_sgb.plot()
store_bond.plot()

''' Calculating Returns '''
last_sgb = df_sgb['Avg_price'].iloc[-1]
last_bond = df_bond['Avg_price'].iloc[-1]
def returns_(x,last_s,last_b):
    if x == 4:
        return_sgb = (store_sgb["Forecasted_value"].iloc[3] - last_s )
        return_bond = ( store_bond["Forecasted_value"].iloc[3] - last_b)
    elif x == 6:
        return_sgb = [(store_sgb["Forecasted_value"].iloc[5]- last_s)]
        return_bond = [(store_bond["Forecasted_value"].iloc[5]- last_b) + 40.5] 
    else:
        return_sgb = [(store_sgb["Forecasted_value"].iloc[11]- last_s) + 85.7725] 
        return_bond = [(store_bond["Forecasted_value"].iloc[11]- last_b) + 81.0]
    return return_sgb, return_bond
gain_sgb,gain_bond = returns_(n,last_sgb,last_bond)

def output_(x,y,t):
    if x > y:
        a = print("The retrun of SGB is {a} and the return of IRFC Bond is {b} after {c} months".format(a=x,b=y,c=t))
    else:
        a = print("The return of IRFC Bond is{a} and the return of SGB Bond is {b} after {c} months".format(a=x,b=y,c=t))
    return a
output_(gain_sgb,gain_bond, n)
```
## Home Page (Used HTML and CSS)
![home](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/Home.png)

## Predict Page
![predict](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/input.png)

## Output Page
![output](https://github.com/Chetan-Bangera/projsgbvsbond/blob/main/img/output.png)

### Project Completed --
