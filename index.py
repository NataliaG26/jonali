from flask import Flask, render_template, abort, request
from flask.wrappers import Request
from model import ARIMA, report, data
import numpy as np
import pandas as pd

app = Flask(__name__)

dataa = data.Data()


@app.route('/')
def home():
    return render_template('layout.html')

##app.add_url_rule('/',view_func=ARIMA.predecir);

@app.route('/horario_compras')
def horarioCompras():
    sales_per_season = report.sales_per_season(dataa.dataMonths)
    sales_per_day = report.sales_per_day(dataa.dataMonths)
    salses_per_weekdays = report.salses_per_weekdays(dataa.dataMonths)
    context = {
         'sales_per_season': sales_per_season,
         'sales_per_day': sales_per_day,
         'salses_per_weekdays' : salses_per_weekdays,
    }
    return render_template('horario_compras.html', **context)

@app.route('/paises')
def paises():
    countryDistr = report.countryDistr(dataa.dataF)
    df = report.countryFreq(dataa.dataF)
    context = {
        'countryDistr' : countryDistr,
        'column_names' : df.columns.values,
        'row_data': list(df.values.tolist()),
        'link_column' : "Country",
        'zip' : zip,
    }
    return render_template('paises.html', **context)


@app.route('/productos_top', methods=['GET', 'POST'])
def productos_top():

    if request.method == 'POST':
        products_num = request.form.get('products_num')
        num = int(products_num)
        most_selled_products, df1 = report.most_selled_products(num, dataa.dataF, dataa.data_products)
        context = {
            'most_selled_products': most_selled_products,
            'column_names' : df1.columns.values,
            'row_data': list(df1.values.tolist()),
            'link_column' : "StockCode",
            'zip' : zip,
        }
        
        return render_template('productos_top.html', **context)
    
    return render_template('productos_top.html')


@app.route('/productos_top_paises', methods=['GET', 'POST'])
def productos_top_paises():

    countries = report.countryFreq(dataa.dataF)['Country'].tolist()
    context = {
        'countries':countries,
    }

    if request.method == 'POST':
        cName = "United Kingdom"
        cName = request.form.get('cSelect')
        products_num = request.form.get('products_num')
        num = int(products_num)
        most_selled_products_per_country, df1 = report.most_selled_products_per_country(num, cName, dataa.dataF, dataa.data_products)
        context = {
            'most_selled_products_per_country': most_selled_products_per_country,
            'column_names' : df1.columns.values,
            'row_data': list(df1.values.tolist()),
            'link_column' : "StockCode",
            'zip' : zip,
            'cName' : cName,
            'countries':countries,
        }
        return render_template('productos_top_paises.html', **context)
    
    return render_template('productos_top_paises.html', **context)


@app.route('/test')
def test():
    return render_template('login.html')


if __name__ == "__main__":
    app.run(debug=True)