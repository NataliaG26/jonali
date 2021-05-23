#Imports Principales
import pandas as pd                #Importamos la librería pandas. Nos va a servir para leer y manipular conjuntos de datos tabulares.
import matplotlib.pyplot as plt    #Importamos pyplot de librería matplotlib. Lo vamos a utilizar para graficar.
import seaborn as sns              #Importamos la librería Seaborn. La vamos a utilizar para graficar.
import numpy as np                 #Importamos la librería numpy para manipular arreglos.
import datetime                    #Importamos datetime para manejar fechas
#MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

def MINIBKMEANS(Self):
    
    results = pd.DataFrame()
    
    def predict(data_customersnotnull):
        #Total de compra
        data_customersnotnull['TotalPrice'] = data_customersnotnull['UnitPrice']*data_customersnotnull['Quantity']

        #fecha actual para el análisis de retención
        today = pd.datetime(2012,1,1)

        #Encontrando los valores de Recency y de Monetary
        df_x = data_customersnotnull.groupby('CustomerID').agg({'TotalPrice': lambda x: x.sum(), #monetary value
                                                                'InvoiceDate': lambda x: (today - x.max()).days}) #recency value 
                                                            #x.max()).days; ultima fecha de compra del cliente

        df_z = data_customersnotnull.groupby('CustomerID').agg({'TotalPrice': lambda x: len(x)}) 
        #Encontrando el valor de frequency per capita

        #Creando la tabla RFM
        rfm_table= pd.merge(df_x,df_z, on='CustomerID')

        #Asignando los nombres de las tablas
        rfm_table.rename(columns= {'InvoiceDate': 'Recency','TotalPrice_y': 'Frequency','TotalPrice_x': 'Monetary'}, inplace= True)

        #Borrar nulls
        rfm_table.dropna(inplace=True)

        #Proceso de conversión / mapeo de variables        
        min_max_scaler = MinMaxScaler((0,1))
        x_scaled = min_max_scaler.fit_transform(rfm_table)
        data_scaled = pd.DataFrame(x_scaled)

        miniBatchKmeans = MiniBatchKMeans(n_clusters=4, init='k-means++', max_iter=100, batch_size=100)
        miniBatchKmeans.fit(data_scaled)
        cluster = miniBatchKmeans.predict(data_scaled)
        result = pd.DataFrame(rfm_table)
        result['cluster_no'] = cluster
        result['cluster_no'].value_counts() #Número total de clientes por clúster
        

        snsplot = sns.set_style("darkgrid")
        snsplot = sns.relplot(data=result2, x="Recency", y="Monetary", hue="cluster_no")        
        name="static/images/MINIBKMEANS.png"
        snsplot.savefig(name)
        
        #cluster average values
        self.results = result2.groupby('cluster_no').mean()

        return name

