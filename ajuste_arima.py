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
    obter_metricas_modelo,
    plot_summary_serie,
)

pd.options.display.float_format = "{:,.2f}".format

df = pd.read_csv("data/LSTM-Multivariate_pollution.csv")

coluna_serie = "pollution"
coluna_tempo = "date"
frequencia_serie = "D"

# fazendo transformações necessárias

df[coluna_tempo] = pd.to_datetime(df[coluna_tempo])
df.set_index(coluna_tempo, inplace=True)
df = df.asfreq(frequencia_serie)

df.describe()

# plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.mean)
# plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.std)

# gera_boxplot(df, coluna_serie, df.index.year)

# gera_acf_pacf(df, coluna_serie, lags=10, tipo="acf")
# gera_acf_pacf(df, coluna_serie, lags=10, tipo="pacf")

#testando a estacionariedade da série

# result = adfuller(df[coluna_serie])
# print("Estatística ADF", result[0])
# print("p-valor", result[1])

p_values = [1, 2, 3]
q_values = [1, 2, 3]
d_values = [0]

train = df.iloc[1:round(len(df)*0.8)]
test = df.iloc[round(len(df)*0.8 + 1): len(df)]

split = TimeSeriesSplit(n_splits=5)

resultados_cv = cross_validate_arimas(p_values, d_values, q_values, df[coluna_serie], split)

resultados_cv

p = 3
d = 0
q = 3

selecionado = ajustar_arima(df[coluna_serie], p, d, q)

print(selecionado.get("warnings"))

gera_graficos_predict(df[coluna_serie], p, d, q)

gera_ljungbox(df[coluna_serie], p, d, q)

gera_diagnosticos(df[coluna_serie], p, d, q)

residuals = selecionado["model"].resid


