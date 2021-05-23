from pandas import pandas as pd 

class Data:

        dataF = pd.DataFrame()
        dataMonths = pd.DataFrame()
        data_products = pd.DataFrame()

        def __init__(self):
                # Vamos a leer el conjunto de datos en un dataframe de pandas. 
                df1 = pd.read_csv('static/data/DB_2009-2010.csv')
                df2 = pd.read_csv('static/data/DB_2010-2011.csv')
                self.dataF = pd.concat([df1, df2]) #Juntar los datos para manejar una sola base de datos
                self.dataF = self.data()
                self.dataMonths = self.dataMonths()
                self.data_products = self.data_products()
                return

        def data(self):
                # Renombrar columnas
                data1 = self.dataF.rename({'Invoice': 'InvoiceNo', 'Price': 'UnitPrice', 'Customer ID': 'CustomerID'}, axis="columns")

                # Cambiar el dtype de las variables que tienen un tipo diferente al que debería
                data1['InvoiceNo'] = data1['InvoiceNo'].astype('category')
                data1['StockCode'] = data1['StockCode'].astype('category')
                data1['Description'] = data1['Description'].astype(str)
                data1['InvoiceDate'] = pd.to_datetime(self.dataF['InvoiceDate'])
                data1['CustomerID'] = data1['CustomerID'].astype('category')
                data1['Country'] = data1['Country'].astype('category')

                # Borrar datos duplicados
                data1 = data1.drop_duplicates()
                
                # Copia de los datos sin Clientes null
                data_without_customersID_null = data1.dropna().copy()
                
                return data1

        def dataMonths(self):
                # Hacer copia de los datos
                dataMonths1 = self.dataF.copy()
                # Agregar una columna con el mes de la compra
                dataMonths1['MonthNo'] = dataMonths1['InvoiceDate'].dt.month
                # Cambiar de numero a nombre del mes
                dataMonths1['Month'] = dataMonths1['MonthNo'].map({1:'Yan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'})
                # Agregar una columna con el año de la compra
                dataMonths1['Year'] = dataMonths1['InvoiceDate'].dt.year
                # Agregar una columna con el día del mes de la compra
                dataMonths1['MonthDay'] = dataMonths1['InvoiceDate'].dt.day
                return dataMonths1

        def data_products(self):
                # Hacer copia de los datos para obtener una tabla solo con los productos
                data_products1 = self.dataF.copy()
                # Extrarer solo StockCode y Description
                data_products1 = data_products1.loc[:, ['StockCode', 'Description']]
                # Borrar datos duplicados
                data_products1 = data_products1.drop_duplicates()
                return data_products1
    