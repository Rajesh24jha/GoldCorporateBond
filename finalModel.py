import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import pickle
import re
from datetime import date
import nsepy
import warnings
warnings.filterwarnings("ignore")

class Forecast_GoldGeneral_Bonnd:
    ##Live data entraction ##
    ##Entracting data by using nsepy package

    todaydate=date.today()
    gold_df= pd.DataFrame(nsepy.get_history(symbol="SGBAUG24",series="GB", start=date(2016,9,1), end=todaydate))
    bond_df= pd.DataFrame(nsepy.get_history(symbol="IRFC",series="N2", start=date(2012,1,1), end=todaydate))

    ##Taking close price as  univariate variable for forecasting
    gold_df=gold_df["Close"].copy()
    gold_df=gold_df.asfreq('D',method='pad')  # Handling Missing weekend and holidays day
    gold_df=gold_df.to_frame()
    
    bond_df=bond_df["Close"].copy()
    bond_df=bond_df.asfreq('D',method='pad')  # Handling Missing weekend and holidays day
    bond_df=bond_df.to_frame()
    
    bond_df.isnull().sum()
    gold_df.isnull().sum()

    bond_df.columns = ["Price"]   # change column name to price
    gold_df.columns = ["Price"]

    ###############################                SQL database                 ################################################3             

    # Connect to the database and convert live data into table SGB and IRFCB and fetch information from table into df_goldbond and df_gebneralbond dataframe
    from sqlalchemy import create_engine
    engine_gold = create_engine('sqlite:///goldbond_database.db', echo=False)
    gold_df.to_sql('SGB', con=engine_gold,if_exists='replace')
    df_goldbond = pd.read_sql('select * from SGB',engine_gold )
    df_goldbond.Date=pd.to_datetime(df_goldbond.Date)
    df_goldbond.set_index('Date',inplace=True)


    engine_bond = create_engine('sqlite:///generalbond_database.db', echo=False)
    bond_df.to_sql('IRFCB', con=engine_bond,if_exists='replace')
    df_generalbond= pd.read_sql('select * from IRFCB',engine_bond)
    df_generalbond.Date=pd.to_datetime(df_generalbond.Date)
    df_generalbond.set_index('Date',inplace=True)
    ###############################                SQL data to python                 ################################################3             

    # check Stationary and adf test
    def test_stationarity(timeseries):
        #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        #Plot rolling statistics:
        fig, an = plt.subplots(figsize=(16, 4))
        an.plot(timeseries, label = "Original Price")
        an.plot(rolmean, label='rolling mean');
        an.plot(rolstd, label='rolling std');
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)
    
        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        print('Test statistic = {:.3f}'.format(adft[0]))
        print('P-value = {:.3f}'.format(adft[1]))
        print('Critical values :')
        for k, v in adft[4].items():
            print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<adft[0] else '', 100-int(k[:-1])))
    

    # Differencing method to remove trend and seasonality
    diff_goldbond = df_goldbond - df_goldbond.shift()
    diff_goldbond.dropna(inplace = True)
   
    diff_generalbond = df_generalbond - df_generalbond.shift()
    diff_generalbond.dropna(inplace = True)
    
    # d = 0

    ''' Finding p, d, q hyperparameter from auto_arima '''
    def arima_pdq(diff_data): ######### change y to df_
        model_autoARIMA = auto_arima(diff_data, start_p=0, start_q=0,
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

    def arima_mod_Gold(self,df_data, n):
        model = ARIMA(df_data, order=(1,0,0), trend = "t") 
        fitted = model.fit()
        
        ''' Forecating '''
        forecasted_val = fitted.predict(1,len(df_data)+(n-1)) 
        forecast_next= fitted.forecast(steps = n, alpha = 0.05) #95 confidence interval
        # Confidence interval of 95%
        forecast = fitted.get_forecast(n, alpha = 0.05)
        conf = forecast.conf_int(alpha=0.05)
        # storing the confidence interval in a series
        lower_series =pd.Series(conf["lower Price"], index=forecasted_val.index[(len(df_data) - 1):])
        upper_series =pd.Series(conf["upper Price"], index=forecasted_val.index[(len(df_data) - 1):])

        Forecast_series_ = pd.concat([lower_series,forecast_next,upper_series], axis = 1)
        Forecast_series_.columns = ["Lower_value","Forecasted_value","Upper_value"]
        return Forecast_series_


       
    def arima_mod_General(self,df_data ,n):
        model = ARIMA(df_data, order=(1,0,1), trend = "t") 
        fitted = model.fit()
        print(fitted.summary())
        ''' Forecating '''
        forecasted_val = fitted.predict(1,len(df_data)+(n-1)) 
        forecast_next= fitted.forecast(steps = n, alpha = 0.05) #95 confidence interval
        # Confidence interval of 95%
        forecast = fitted.get_forecast(n, alpha = 0.05)
        conf = forecast.conf_int(alpha=0.05)
        # storing the confidence interval in a series
        lower_series =pd.Series(conf["lower Price"], index=forecasted_val.index[(len(df_data) - 1):])
        upper_series =pd.Series(conf["upper Price"], index=forecasted_val.index[(len(df_data) - 1):])

        Forecast_series_ = pd.concat([lower_series,forecast_next,upper_series], axis = 1)
        Forecast_series_.columns = ["Lower_value","Forecasted_value","Upper_value"]
        return Forecast_series_

# End of class
   


   
forecast_object=Forecast_GoldGeneral_Bonnd()
my_pickled_object = pickle.dumps(forecast_object)  # Pickling the object

#print(f"This is my pickled object:\n{my_pickled_object}\n")

#store_sgb =forecast_object.arima_mod_Gold(forecast_object.df_goldbond,3)

