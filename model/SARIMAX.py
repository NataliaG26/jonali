#Imports Principales
import pandas as pd                #Importamos la librería pandas. Nos va a servir para leer y manipular conjuntos de datos tabulares.
import matplotlib.pyplot as plt    #Importamos pyplot de librería matplotlib. Lo vamos a utilizar para graficar.
import numpy as np                 #Importamos la librería numpy para manipular arreglos.
import datetime                    #Importamos datetime para manejar fechas
#SARIMAX
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from numpy import log
import warnings 
warnings.filterwarnings("ignore")
#Métricas
from math import sqrt #Permite usar la función de raíz


def SARIMAX(self):

    results = pd.DataFrame()

    def Predict(dataMonths):
        # Separar los datos por año
        df_2010 = dataMonths[dataMonths.Year.isin([2010])]
        df_2011 = dataMonths[dataMonths.Year.isin([2011])]
        # Contar las ventas por mes en cada año
        monthFreq_2010 = df_2010.groupby(['MonthNo']).count() 
        monthFreq_2011 = df_2011.groupby(['MonthNo']).count()

        #Copias de las variables de frecuencia usadas para el primer gráfico de resumen
        Ventas_2010 = monthFreq_2010.copy()
        Ventas_2011 = monthFreq_2011.copy()

        #Se borra todo lo que no es de interés para este análisis y se deja lo importante, Mes y numero de ventas para ese mes
        Ventas_2010.drop(['InvoiceNo','StockCode','Description','InvoiceDate','UnitPrice','CustomerID','Country','Month','Year'],axis=1,inplace=True)
        Ventas_2011.drop(['InvoiceNo','StockCode','Description','InvoiceDate','UnitPrice','CustomerID','Country','Month','Year'],axis=1,inplace=True)

        #Se juntan todos los dataframe en uno solo
        ventas = pd.concat([Ventas_2010,Ventas_2011])

        #Se listan todas las fechas y se asume el número de ventas reportado al final del mes
        ventas['Month'] = ['01/31/2010','02/28/2010','03/31/2010','04/30/2010','05/31/2010','06/30/2010','07/31/2010','08/31/2010','09/30/2010','10/31/2010','11/30/2010','12/31/2010',
                            '01/31/2011','02/28/2011','03/31/2011','04/30/2011','05/31/2011','06/01/2011','07/31/2011','08/31/2011','09/30/2011','10/31/2011','11/30/2011','12/31/2011',]
        ventas['Month'] = pd.to_datetime(ventas["Month"])
        ventas.set_index(['Month'],inplace=True)
        ventas.index = pd.DatetimeIndex(ventas.index).to_period('M')
        ventas.columns = ['Sales']

        # Split data into train / test sets
        train = ventas.iloc[:len(ventas)-12] 
        test = ventas.iloc[len(ventas)-12:] # set one year(12 months) for testing

        # Fit a SARIMAX(0, 1, 0, 12) on the training set
        model = SARIMAX(train['Sales'],order=(1,0,1),seasonal_order =(0, 1, 0, 12))
        result = model.fit()    

        #Aquí se utiliza el conjunto de test indirectamente teniendo en cuenta los proximos meses a predecir
        start = len(train)
        end = len(train) + len(test) - 1
        
        # Prediccciones para un año en frente con el conjunto de test
        predictions = result.predict(start, end, typ = 'levels').rename("Predictions")
        
        self.results = predictions.to_frame().copy()
        testcopy = test.copy()
        testcopy['Predictions'] = results['Predictions']

        # plot predictions and actual values
        predictions.plot(legend = True)
        test['Sales'].plot(legend = True)
        name = "static/images/SARIMAX.png"
        plt.savefig(name, dpi=300)
        
        return name

        