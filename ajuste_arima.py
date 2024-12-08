import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller

from funcoes import (
    ajustar_arima,
    cross_validate_arimas,
    gera_acf_pacf,
    gera_boxplot,
    gera_diagnosticos,
    gera_graficos_predict,
    gera_ljungbox,
    plot_summary_serie,
)

pd.options.display.float_format = "{:,.2f}".format

df_train = pd.read_csv("data/LSTM-Multivariate_pollution.csv")

df_test = pd.read_csv("data/pollution_test_data1.csv")

coluna_serie = "pollution"
coluna_tempo = "date"
frequencia_serie = "h"

# fazendo transformações necessárias

df_train[coluna_tempo] = pd.to_datetime(df_train[coluna_tempo])
df_train.set_index(coluna_tempo, inplace=True)
df_train = df_train.asfreq(frequencia_serie)

df_train.describe()

# plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.mean)
# plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.std)

# gera_boxplot(df, coluna_serie, df.index.year)

# gera_acf_pacf(df, coluna_serie, lags=10, tipo="acf")
# gera_acf_pacf(df, coluna_serie, lags=10, tipo="pacf")

# #testando a estacionariedade da série

# result = adfuller(df[coluna_serie])
# print("Estatística ADF", result[0])
# print("p-valor", result[1])

# p_values = [1, 2, 3]
# q_values = [1, 2, 3]
# d_values = [0]

# split = TimeSeriesSplit(n_splits=2)

# resultados_cv = cross_validate_arimas(p_values, d_values, q_values, df[coluna_serie], split)

# resultados_cv

p = 1
d = 0
q = 1

selecionado = ajustar_arima(df_train[coluna_serie], p, d, q)

print(selecionado.get("warnings"))

gera_graficos_predict(df_train[coluna_serie], p, d, q)

gera_ljungbox(df_train[coluna_serie], p, d, q)

gera_diagnosticos(df_train[coluna_serie], p, d, q)

residuals_train = selecionado["model"].resid

yhat = gera_graficos_predict(df_test[coluna_serie], p, d, q)

residuals_test = df_test[coluna_serie] - yhat

