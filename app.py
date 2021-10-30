#Flask imports
from flask import Flask, request,render_template,jsonify
import pickle
from finalModel import my_pickled_object

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from io import BytesIO
import base64



pkl = pickle.loads(my_pickled_object)  # Unpickling the object
#print(f"This is a_dict of the unpickled object:\n{pkl.todaydate}\n")

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('Home.html')

@app.route("/Output", methods=("POST", "GET"))
def Output():
    # HTML -> .py
    if request.method == "POST":
        n = int(request.form["Time Period"])
        store_goldbond = pkl.arima_mod_Gold(pkl.df_goldbond,n*365)
        store_generalbond = pkl.arima_mod_General(pkl.df_generalbond,n*365)
    
    ''' Calculating Returns '''
    last_goldbond = pkl.df_goldbond['Price'].iloc[-1]
    last_generalbond = pkl.df_generalbond['Price'].iloc[-1] 
    
    # Combining Forecasting Value of both Gold and General Bond 
    forecast_df=store_goldbond["Forecasted_value"]
    forecast_df=forecast_df.to_frame()
    forecast_df["General_Bond"]=store_generalbond["Forecasted_value"]
    forecast_df.rename(columns = {'Forecasted_value':'Gold_Bond'}, inplace = True)
    forecast_df=forecast_df.round(2)

    img = BytesIO()
    forecast_df1 =  forecast_df.cumsum()
    plt.figure()
    forecast_df1.plot()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    def returns_(n,last_gold,last_bond):
        if n== 3:
            return_sgb = (store_goldbond["Forecasted_value"].iloc[(n*365)-1] - last_gold )
            return_bond = ( store_generalbond["Forecasted_value"].iloc[(n*365)-1] - last_bond)
        elif n == 5:
            return_sgb = [(store_goldbond["Forecasted_value"].iloc[(n*365)-1]- last_gold)]
            return_bond = [(store_generalbond["Forecasted_value"].iloc[(n*365)-1]- last_bond)] 
        else:
            return_sgb = [(store_goldbond["Forecasted_value"].iloc[(n*365)-1]- last_gold)] 
            return_bond = [(store_generalbond["Forecasted_value"].iloc[(n*365)-1]- last_bond)]
        return return_sgb, return_bond

    gain_goldbond,gain_generalbond = returns_(n,last_goldbond,last_generalbond)
    goldper= (gain_goldbond / last_goldbond)*100
    
    bondper=(gain_generalbond / last_generalbond) * 100
    
    def output_(x,y):
        if x > y:
            return ("The gain of Gold Bond is higher than of General Bond by : " +str(x) +" %")
        else:
            return ("The gain of General Bond is higher than of Gold Bond by : " +str(y)+" %")

    prt = str(output_(goldper,bondper))
    
    # .py -> HTML
    return render_template('Output.html',tables= [forecast_df.to_html(classes='data')],titles=['na','Gold and General Bond Forecasting Data:'],result1=prt,plot_url=plot_url)
    
   
if __name__ == "__main__":
    app.run(debug=True)


