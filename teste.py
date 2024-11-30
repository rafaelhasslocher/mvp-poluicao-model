import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
import warnings

pd.options.display.float_format = "{:,.2f}".format

df = pd.read_csv("data/LSTM-Multivariate_pollution.csv")

# fazendo transformações necessárias

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.asfreq("D")

# fazendo o resampling para considerar a poluição média por mês

df_mensal_media = df.resample("ME")["pollution"].mean().reset_index().copy()
df_mensal_variancia = df.resample("ME")["pollution"].var().reset_index().copy()

df_mensal_media.set_index("date", inplace=True)
df_mensal_variancia.set_index("date", inplace=True)

tela, (media, variancia) = plt.subplots(2, figsize=(10, 10))
tela.patch.set_facecolor("whitesmoke")
media.plot(df_mensal_media.index, df_mensal_media["pollution"])
media.set_title("Poluição média mensal")
media.set_xlabel("Data")
media.set_ylabel("Poluição")

variancia.plot(df_mensal_variancia.index, df_mensal_variancia["pollution"])
variancia.set_title("Desvio padrão da poluição mensal")
variancia.set_xlabel("Data")
variancia.set_ylabel("Poluição")

plt.show()

# testando a estacionariedade da série

result = adfuller(df["pollution"])
print("ADF Statistic", result[0])
print("p-valor", result[1])

plot_acf(df["pollution"], lags=80)
plt.ylim(0, 1)
plt.xlim(0.7, 10)

plot_pacf(df["pollution"])
plt.ylim(0, 1)
plt.xlim(0.7, 10)

# Supondo que df tenha uma coluna 'date' no formato datetime
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.asfreq("D")


serie = df["pollution"]

p_values = [1, 2, 3]
q_values = [1, 2, 3]

params = {}

for count, (p, q) in enumerate(product(p_values, q_values)):
    params[f"model_{count}"] = {"p": p, "q": q}

resultados = {}

for name, param in params.items():
    try:
        with warnings.catch_warnings(record=True) as warns:
            model = ARIMA(serie, order=(param["p"], 0, param["q"]))
            model_fit = model.fit()
        resultados[name] = {
            **param,
            "model": model_fit,
            "warnings": [warn.message for warn in warns],
        }
    except Exception as e:
        print(f"Erro ao ajustar o modelo ARIMA({p}, 0, {q}): {e}")

#incluir uma função para receber um model_fit e gerar os gráficos

# for p in p_values:
#     for q in q_values:
#         try:
#             # Ajustar o modelo ARIMA
#             model = ARIMA(serie, order=(p, 0, q))
#             model_fit = model.fit()

#             # Fazer previsões
#             y_hat = model_fit.predict(start=0, end=len(serie) - 1)

#             # Plotar os valores observados e previstos
#             import matplotlib.pyplot as plt

#             plt.figure(figsize=(10, 6))
#             plt.plot(serie, label="Valores Observados")
#             plt.plot(y_hat, label=f"Previsões ARIMA({p}, 0, {q})", color="red")
#             plt.xlabel("Data")
#             plt.ylabel("Poluição")
#             plt.title(f"Valores Observados e Previstos usando ARIMA({p}, 0, {q})")
#             plt.legend()
#             plt.show()
#             print(model_fit.aic)
#         except Exception as e:
#             print(f"Erro ao ajustar o modelo ARIMA({p}, 0, {q}): {e}")


plt.figure(figsize=(10, 6))
plt.plot(df["pollution"], label="Série Temporal")
plt.plot(y_hat, label="Previsão", color="red")
plt.xlabel("Data")
plt.ylabel("Poluição")
plt.title("Previsão de Poluição usando ARIMA")
plt.legend()
plt.show()

residuos = model_fit.resid  # Plotar os resíduos
plt.figure(figsize=(10, 6))
plt.scatter(residuos.index, residuos)
plt.xlabel("Data")
plt.ylabel("Resíduos")
plt.title("Resíduos do Modelo ARIMA")
plt.show()

# Realizar o teste de Ljung-Box
ljung_box_result = acorr_ljungbox(residuos, lags=[10], return_df=True)
print(ljung_box_result)