import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

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

# fazendo o resampling para considerar a poluição média por mês

df_mensal_media = df.resample("ME")[coluna_serie].mean().reset_index().copy()
df_mensal_variancia = df.resample("ME")[coluna_serie].std().reset_index().copy()

df_mensal_media.set_index(coluna_tempo, inplace=True)
df_mensal_variancia.set_index(coluna_tempo, inplace=True)

# Criar a figura e os eixos
tela, (y, media, variancia) = plt.subplots(3, 1, figsize=(15, 15))
tela.patch.set_facecolor("whitesmoke")

# Plotar o gráfico da série temporal
sns.lineplot(ax=y, x=df.index, y=df[coluna_serie])
y.set_title("Desvio padrão da poluição mensal")
y.set_xlabel("Data")
y.set_ylabel("Poluição")  # noqa: E703


# Plotar a poluição média mensal
sns.lineplot(ax=media, x=df_mensal_media.index, y=df_mensal_media[coluna_serie])
media.set_title("Poluição média mensal")
media.set_xlabel("Data")
media.set_ylabel("Poluição")  # noqa: E703

# Plotar o desvio padrão da poluição mensal
sns.lineplot(
    ax=variancia, x=df_mensal_variancia.index, y=df_mensal_variancia[coluna_serie]
)
variancia.set_title("Desvio padrão da poluição mensal")
variancia.set_xlabel("Data")
variancia.set_ylabel("Poluição")  # noqa: E703

tela, boxplot = plt.subplots(1, 1, figsize=(7, 7))
tela.patch.set_facecolor("whitesmoke")

sns.boxplot(
    x=df.index.year,
    y=df[coluna_serie],
    ax=boxplot,
    hue=df.index.year,
    palette="GnBu",
    legend=False,
)
boxplot.set_title("Box-plot da poluição anual", fontsize=15)
boxplot.set_xlabel("Ano", fontsize=10)
boxplot.set_ylabel("Poluição", fontsize=10)  # noqa: E703
boxplot.tick_params(axis="x", labelsize=8)
boxplot.tick_params(axis="y", labelsize=8)

plt.show()

# testando a estacionariedade da série

result = adfuller(df[coluna_serie])
print("Estatística ADF", result[0])
print("p-valor", result[1])

# criar figura e eixos

sns.set_theme(style="whitegrid")

tela, (acf, pacf) = plt.subplots(2, 1, figsize=(10, 10))
tela.patch.set_facecolor("whitesmoke")

plot_acf(df[coluna_serie], lags=10, ax=acf)
acf.set_ylim(0, 1)
acf.set_xlim(0.7, 5.5)
acf.set_xticks(np.arange(1, 5.5))
acf.set_title("Gráfico de Autocorrelação")  # noqa: E703

plot_pacf(df[coluna_serie], ax=pacf)
pacf.set_ylim(0, 1)
pacf.set_xlim(0.7, 5.5)
pacf.set_xticks(np.arange(1, 5.5))
pacf.set_title("Gráfico de Autocorrelação Parcial")  # noqa: E703


p_values = [1, 2, 3]
q_values = [1, 2, 3]

params = {}

for count, (p, q) in enumerate(product(p_values, q_values)):
    params[f"model_{count}"] = {"p": p, "q": q}

resultados = {}

for name, param in params.items():
    try:
        with warnings.catch_warnings(record=True) as warns:
            model = ARIMA(df[coluna_serie], order=(param["p"], 0, param["q"]))
            model_fit = model.fit()
        resultados[name] = {
            **param,
            "model": model,
            "warnings": [warn.message for warn in warns],
            "AIC": model_fit.aic,
            "BIC": model_fit.bic,
        }
    except Exception as e:
        print(f"Erro ao ajustar o modelo ARIMA({p}, 0, {q}): {e}")


def gera_graficos(df, resultados):
    aic_bic = {}
    tela, axs = plt.subplots(len(resultados), 1, figsize=(10, 5 * (len(resultados))))
    for ax, (nome_modelo, modelo) in zip(axs, resultados.items()):
        # Exemplo de plotagem (substitua com seu código de plotagem)

        model_fit = modelo["model"]
        yhat = model_fit.predict(start=0, end=len(df[coluna_serie]) - 1)
        sns.lineplot(ax=ax, x=df.index, y=df[coluna_serie])
        sns.lineplot(ax=ax, x=df.index, y=yhat, color="red")
        ax.set_title(f"Gráfico para {nome_modelo}")
        ax.set_xlabel("Eixo X")
        ax.set_ylabel("Eixo Y")

        aic_bic[nome_modelo] = {"AIC": model_fit.aic, "BIC": model_fit.bic}

    plt.show()

    return aic_bic


aic_bic = pd.DataFrame(
    [
        (modelo, nome_modelo["AIC"], nome_modelo["BIC"])
        for modelo, nome_modelo in resultados.items()
    ],
    columns=["Modelo", "AIC", "BIC"]
)

modelo_maximo_aic = aic_bic['Modelo'].loc[aic_bic['AIC'].idxmax()]
modelo_maximo_bic = aic_bic['Modelo'].loc[aic_bic['AIC'].idxmax()]

p_maximo_aic = resultados[modelo_maximo_aic]["p"]
q_maximo_aic = resultados[modelo_maximo_aic]["q"]

p_maximo_bic = resultados[modelo_maximo_bic]["p"]
q_maximo_bic = resultados[modelo_maximo_bic]["q"]

print(rf"O modelo com maior AIC foi o ARIMA({p_maximo_aic}, 0, {q_maximo_bic})"
       rf" e o de maior BIC foi o ARIMA({p_maximo_bic}, 0, {q_maximo_bic}).")


###

residuos = model_fit.resid
# Realizar o teste de Ljung-Box
ljung_box_result = acorr_ljungbox(residuos, lags=[10], return_df=True)
print(ljung_box_result)

# incluir uma função para receber um model_fit e gerar os gráficos

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


