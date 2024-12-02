import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    params[f"Modelo_{count}"] = {"p": p, "q": q}

resultados = {}

for name, param in params.items():
    try:
        with warnings.catch_warnings(record=True) as warns:
            model = ARIMA(df[coluna_serie], order=(param["p"], 0, param["q"]))
            model_fit = model.fit()
        resultados[name] = {
            **param,
            "Modelo": model_fit,
            "warnings": [warn.message for warn in warns],
            "AIC": model_fit.aic,
            "BIC": model_fit.bic,
        }
    except Exception as e:
        print(f"Erro ao ajustar o modelo ARIMA({p}, 0, {q}): {e}")


def gera_graficos_predict(model_fit, p, q):
    tela_yhat = plt.figure(figsize=(10, 8))

    tela_yhat.patch.set_facecolor("whitesmoke")

    grafico_yhat = sns.lineplot(x=df.index, y=df[coluna_serie], label="Real")
    sns.lineplot(x=df.index, y=yhat, color="red", label="Previsto")
    grafico_yhat.set_title(f"Gráfico para ARIMA({p}, 0, {q})")
    grafico_yhat.set_xlabel("Eixo X")
    grafico_yhat.set_ylabel("Eixo Y")
    
    return yhat
    
def gera_ljungbox(model_fit):
    
    ljungbox = model_fit.test_serial_correlation(method="ljungbox")
    
    print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, 0, {q}):\n")
    
    for lag in range(len(ljungbox[0][0])):
        p_valor = ljungbox[0][1][lag]
        print(f"Lag {lag + 1}: p-valor = {p_valor}")

for i, (nome_modelo, modelo) in enumerate(resultados.items()):
    
    p = modelo['p']
    q = modelo['q']
    model_fit = modelo["Modelo"]
    
    yhat = model_fit.predict(start=0, end=len(df[coluna_serie]) - 1)

    predict = gera_graficos_predict(yhat, p, q)
    
    ljungbox = gera_ljungbox(model_fit)

    tela_diagnostics = plt.figure(figsize=(10, 8))
    tela_diagnostics.patch.set_facecolor("whitesmoke")
    model_fit.plot_diagnostics(fig=tela_diagnostics)
    tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p}, 0, {q})")











def gera_analises(df, resultados, coluna_serie):
    aic_bic = {}
    tela_yhat, previstos = plt.subplots(
        len(resultados), 1, figsize=(10, 5 * len(resultados))
    )
    tela_yhat.patch.set_facecolor("whitesmoke")
    for i, (nome_modelo, modelo) in enumerate(resultados.items()):
        p = modelo['p']
        q = modelo['q']
        model_fit = modelo["Modelo"]

        yhat = model_fit.predict(start=0, end=len(df[coluna_serie]) - 1)

        sns.lineplot(ax=previstos[i], x=df.index, y=df[coluna_serie], label="Real")
        sns.lineplot(ax=previstos[i], x=df.index, y=yhat, color="red", label="Previsto")
        previstos[i].set_title(f"Gráfico para ARIMA({p}, 0, {q})")
        previstos[i].set_xlabel("Eixo X")
        previstos[i].set_ylabel("Eixo Y")
        plt.close(tela_yhat)

        ljungbox = model_fit.test_serial_correlation(method='ljungbox')

        mse = mean_squared_error(df[coluna_serie], yhat)
        rmse = mse ** 0.5
        print(f"\nAcurácia do modelo ARIMA({p}, 0, {q}): RMSE = {rmse}")

        print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, 0, {q}):\n")

        for lag in range(len(ljungbox[0][0])):
            p_valor = ljungbox[0][1][lag]
            print(f"Lag {lag + 1}: p-valor = {p_valor}")

        tela_diagnostics = plt.figure(figsize=(10, 8))
        tela_diagnostics.patch.set_facecolor("whitesmoke")
        model_fit.plot_diagnostics(fig=tela_diagnostics)
        tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p}, 0, {q})")

        aic_bic[(p, q)] = {"AIC": model_fit.aic, "BIC": model_fit.bic}

        plt.show()
        plt.close(tela_diagnostics)

    aic_bic = pd.DataFrame(
        [
            (modelo, nome_modelo["AIC"], nome_modelo["BIC"])
            for modelo, nome_modelo in resultados.items()
        ],
        columns=["Modelo", "AIC", "BIC"],
    )

    modelo_maximo_aic = aic_bic["Modelo"].loc[aic_bic["AIC"].idxmax()]
    modelo_maximo_bic = aic_bic["Modelo"].loc[aic_bic["AIC"].idxmax()]

    p_maximo_aic = resultados[modelo_maximo_aic]["p"]
    q_maximo_aic = resultados[modelo_maximo_aic]["q"]

    p_maximo_bic = resultados[modelo_maximo_bic]["p"]
    q_maximo_bic = resultados[modelo_maximo_bic]["q"]

    print(
        rf"O modelo com maior AIC foi o ARIMA({p_maximo_aic}, 0, {q_maximo_aic})"
        rf" e o de maior BIC foi o ARIMA({p_maximo_bic}, 0, {q_maximo_bic})."
    )

    return aic_bic

aic_bic = gera_analises(df, resultados, coluna_serie)
