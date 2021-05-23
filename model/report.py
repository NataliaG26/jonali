
import pandas as pd                #Importamos la librería pandas. Nos va a servir para leer y manipular conjuntos de datos tabulares.
from matplotlib import pyplot as plt    #Importamos pyplot de librería matplotlib. Lo vamos a utilizar para graficar.
import numpy as np
from copy import deepcopy      #Permite hacer copias profundas. 

# Lista de paises que compran en la tienda
def countryFreq(data):
    countryFreq = data.groupby(['Country']).count() 
    countryFreq = countryFreq.sort_values('InvoiceNo', ascending= False)
    countryFreq = countryFreq.rename_axis('Country').reset_index()
    countryFreq = countryFreq.loc[:, ['Country', 'InvoiceNo']]
    return countryFreq

# Grafica con la distribución de compras por país
def countryDistr(data):
    countryValue = data['Country'].value_counts()
    others = countryValue.sum() - countryValue['United Kingdom']
    dfCountry = pd.DataFrame({'Country': ['United Kingdom', 'Otros'],
                   'Country': [countryValue['United Kingdom'], others]},
                   index=['United Kingdom', 'Otros'])
    plot = dfCountry.plot.pie(y='Country', figsize=(11, 6),
                                      #autopct='%1.1f%%',  #Esto muestra úicamente el porcentaje de ejemplos de cada categoría en el gráfico. 
                                      autopct=lambda pct: '{:.1f}%\n({:d})'.format(pct, int(pct/100.*countryValue.sum())), #Con esta función lambda se puede mostrar el porcentaje y el número absoluto de ejemplos en cada categoría. 
                                      startangle=0,       #Esto define la orientación de la línea vertical inicial de las divisiones del pie chart.
                                      fontsize=14,         #Tamaño de los textos.         
                                      cmap='Pastel1')
    name = "static/images/Distribucion_compras_pais.png"
    plt.savefig(name, dpi=300)
    return name

def sales_per_season(dataMonths):
    # Separar los datos por año
    df_2009 = dataMonths[dataMonths.Year.isin([2009])]
    df_2010 = dataMonths[dataMonths.Year.isin([2010])]
    df_2011 = dataMonths[dataMonths.Year.isin([2011])]
    # Contar las ventas por mes en cada año
    monthFreq_2009 = df_2009.groupby(['MonthNo']).count()
    monthFreq_2010 = df_2010.groupby(['MonthNo']).count() 
    monthFreq_2011 = df_2011.groupby(['MonthNo']).count()
    # Lista de los nombres de los meses
    listMonths = ['Yan','Feb','Mar','Apr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec']
    # Convertir datos a lista
    list_2009 = monthFreq_2009['InvoiceNo'].tolist()
    list_2010 = monthFreq_2010['InvoiceNo'].tolist()
    list_2011 = monthFreq_2011['InvoiceNo'].tolist()

    # Mostar gráfica
    fig, ax = plt.subplots()
    ax.plot(listMonths, list_2009)
    ax.plot(listMonths, list_2010)
    ax.plot(listMonths, list_2011)
    ax.set(xlabel='Mes', ylabel='Ventas',
           title='Ventas de cada mes por año')
    ax.grid()
    ax.plot(list_2009, label = "2009", color='b')
    ax.plot(list_2010, label = "2010", color='g')
    ax.plot(list_2011, label = "2011", color='r')
    ax.legend()
    ax.legend(loc="upper left")
    name = "static/images/ventas_mes_anio.png"
    fig.savefig(name)
    return name

def sales_per_day(dataMonths):
    # Contar las ventas por año
    yearFreq = dataMonths.groupby(['Year']).count()
    # Separar los datos por año
    df_2009 = dataMonths[dataMonths.Year.isin([2009])]
    df_2010 = dataMonths[dataMonths.Year.isin([2010])]
    df_2011 = dataMonths[dataMonths.Year.isin([2011])]
    # Contar las ventas por día del mes en cada año
    monthFreq_2009 = df_2009.groupby(['MonthDay']).count()
    monthFreq_2010 = df_2010.groupby(['MonthDay']).count() 
    monthFreq_2011 = df_2011.groupby(['MonthDay']).count()
    # Convertir datos a lista
    list_2009 = monthFreq_2009['InvoiceNo'].tolist()
    list_2010 = monthFreq_2010['InvoiceNo'].tolist()
    list_2011 = monthFreq_2011['InvoiceNo'].tolist()

    # Mostar gráfica
    fig, ax = plt.subplots()
    ax.plot(list_2009)
    ax.plot(list_2010)
    ax.plot(list_2011)
    ax.set(xlabel='Día', ylabel='Ventas',
           title='Ventas de cada día del mes por año')
    ax.grid()
    ax.plot(list_2009, label = "2009", color='b')
    ax.plot(list_2010, label = "2010", color='g')
    ax.plot(list_2011, label = "2011", color='r')
    ax.legend()
    ax.legend(loc="upper left")
    plt.yticks(np.arange(0,40000,5000))
    plt.xticks(np.arange(0,32,2))
    name = "static/images/ventas_dia_mes_anio.png"
    fig.savefig(name)
    return name

def salses_per_weekdays(dataMonths):
    # Hacer una copia de los datos
    dataWeekDays = dataMonths.copy()
    # Agregar una columna con el día de la semana que se realiza la compra
    dataWeekDays['weekdayNo'] = dataWeekDays['InvoiceDate'].dt.dayofweek
    # Agregar una columna con el nombre del día de la semana que realiza la compara
    dataWeekDays['weekday'] = dataWeekDays['weekdayNo'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thru', 4:'Fri', 5:'Sat', 6:'Sun'})
    # Cuenta las compras realizadas en cada día de la semana
    dayFreq = dataWeekDays.groupby(['weekdayNo']).count()
    # Crear una lista con los datos por cada año
    df_2009 = dataWeekDays[dataWeekDays.Year.isin([2009])]
    df_2010 = dataWeekDays[dataWeekDays.Year.isin([2010])]
    df_2011 = dataWeekDays[dataWeekDays.Year.isin([2011])]
    # Contar las compras por día de la semana en cada año
    weekdayFreq_2009 = df_2009.groupby(['weekdayNo']).count()
    weekdayFreq_2010 = df_2010.groupby(['weekdayNo']).count() 
    weekdayFreq_2011 = df_2011.groupby(['weekdayNo']).count() 
    # Lista de los nombres de los días de la semana
    listDays = ['Mon','Tue','Wed','Thru','Fri','Sat','Sun']
    # Convertir a listas
    list_2009 = weekdayFreq_2009['InvoiceNo'].tolist()
    list_2010 = weekdayFreq_2010['InvoiceNo'].tolist()
    list_2011 = weekdayFreq_2011['InvoiceNo'].tolist()

    # Crear y mostrar lista
    fig, ax = plt.subplots()
    ax.plot(listDays, list_2009)
    ax.plot(listDays, list_2010)
    ax.plot(listDays, list_2011)
    ax.set(xlabel='Mes', ylabel='Ventas',
           title='Ventas de cada día de la semana por año')
    ax.grid()
    ax.plot(list_2009, label = "2009", color='b')
    ax.plot(list_2010, label = "2010", color='g')
    ax.plot(list_2011, label = "2011", color='r')
    ax.legend()
    ax.legend(loc="center left")
    name = "static/images/ventas_semana_anio.png"
    fig.savefig(name)
    return name

#Productos más vendidos
def most_selled_products(num, data, data_products):
    # Hacer una copia de los datos
    dataProd = data.copy()
    # Agrupas y contar los datos por producto
    prodFreq = dataProd.groupby(['StockCode']).count()
    # Organizar de mayor a menor número de compras
    prodFreq = prodFreq.sort_values('InvoiceNo', ascending= False)
    # Reset index
    prodFreq = prodFreq.rename_axis('StockCode').reset_index()
    # Merge
    # Macer el merge entre el conteo y la tabla con la descripción
    prodFreq = pd.merge(prodFreq, data_products, how='left', on='StockCode')
    # Borrar datos duplicados que resultaron del merge
    prodFreq = prodFreq.drop_duplicates(subset=['StockCode'])
    prodFreq2 = prodFreq.reset_index()
    prodFreq2 =  prodFreq2.loc[0:num , ['StockCode', 'Description_y', 'InvoiceNo']]
    
    # Crear y mostrar gráfica
    stockCodeList = prodFreq['Description_y'].tolist()[0:num]
    x1 = prodFreq['InvoiceNo'].tolist()[0:num]
    fig, ax = plt.subplots()
    ax.set(xlabel='Ventas', ylabel='Productos',
           title='Productos más vendidos')
    ax.barh(stockCodeList, x1, label = "General")
    name = "static/images/top_prod.png"
    fig.savefig(name)
    return name, prodFreq2

# Conteo de los productos vendidos en un pais, ordenados de mayor a menor
def product_country_data(name, data):
    country_data = data[data.Country == name]
    # Agrupar y contar los datos por productos
    country_data_count = country_data.groupby(['StockCode']).count()
    # Organizar los datos de mayor a menor
    country_data_count = country_data_count.sort_values('InvoiceNo', ascending = False)
    return country_data_count

# Productos más vendidos en cada pais
def most_selled_products_per_country(num, cName, data, data_products):
    # Conteo de los productos vendidos en un pais
    country_data_count = product_country_data(cName, data)
    # Borrar de la lista los productos que no se venden (Conteo == 0)
    country_data_count = country_data_count.drop(country_data_count[country_data_count['InvoiceNo'] == 0].index)

    # -------------------------------- Gráfica ---------------------------------
    # Macer el merge entre el conteo y la tabla con la descripción
    country_data_count2 = pd.merge(country_data_count, data_products, how='left', on='StockCode')
    # Borrar datos duplicados que resultaron del merge
    country_data_count2 = country_data_count2.drop_duplicates(subset=['StockCode'])
    country_data_count3 = country_data_count2.reset_index()
    country_data_count3 =  country_data_count2.loc[:, ['StockCode', 'Description_y', 'InvoiceNo']]
    country_data_count3 = country_data_count3.iloc[0:num]
    
    # Crear y mostrar gráfica
    country_data_stockList = country_data_count2['Description_y'].tolist()[0:num]
    x1 = country_data_count2['InvoiceNo'].tolist()[0:num]
    fig, ax = plt.subplots()
    ax.set(xlabel='Ventas', ylabel='Productos',
           title='Productos más vendidos en '+cName)
    ax.barh(country_data_stockList, x1, label = "General")
    name = "static/images/top_prod_country.png"
    fig.savefig(name)
    return name, country_data_count3