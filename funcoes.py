import warnings
from collections import Counter
from itertools import product
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


def plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade, summary):
    df = df.resample(periodicidade)[coluna_serie].apply(summary).reset_index()
    df.set_index(coluna_tempo, inplace=True)

    plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    grafico_summary = sns.lineplot(x=df.index, y=df[coluna_serie])

    summary_name = str(summary.__name__)

    grafico_summary.set_title(
        f"Sumarização considerando a função {summary_name} e periodicidade {periodicidade}"
    )

    plt.show()


def gera_boxplot(df, coluna_serie, index):
    boxplot = plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    ax = boxplot.add_subplot(111)

    plot = sns.boxplot(
        x=index,
        y=df[coluna_serie],
        ax=ax,
        hue=index,
        palette="GnBu",
        legend=False,
    )

    plot.set_title("Box-plot da poluição anual", fontsize=15)
    plot.set_xlabel("Ano", fontsize=10)
    plot.set_ylabel("Poluição", fontsize=10)  # noqa: E703
    plot.tick_params(axis="x", labelsize=8)
    plot.tick_params(axis="y", labelsize=8)

    plt.show()


def gera_acf_pacf(df, coluna_serie, lags, tipo):
    if tipo == "acf":
        acf_values = acf(df[coluna_serie], nlags=lags)
        plot_acf(df[coluna_serie], lags=lags)
        plt.title("Gráfico de Autocorrelação")
        plt.tick_params(labelsize=7)

        plt.ylim(bottom=min(0, min(acf_values)), top=max(acf_values))
    else:
        pacf_values = pacf(df[coluna_serie], nlags=lags)
        plot_pacf(df[coluna_serie], lags=lags)
        plt.title("Gráfico de Autocorrelação parcial")
        plt.tick_params(labelsize=7)

        plt.ylim(bottom=min(0, min(pacf_values)), top=max(pacf_values))

    plt.gcf().set_facecolor("whitesmoke")
    plt.show()


def count_warnings(warnings_list):
    tipos_warnings = {}
    for warn in warnings_list:
        tipo = type(warn.message).__name__
        if tipo in tipos_warnings:
            tipos_warnings[tipo] += 1
        else:
            tipos_warnings[tipo] = 1
    return tipos_warnings


def acumula_warnings(tipos_warnings):
    warnings_acumulados = {}
    for tipo, quantidade in tipos_warnings.items():
        if tipo in warnings_acumulados:
            warnings_acumulados[tipo] += quantidade
        else:
            warnings_acumulados[tipo] = quantidade
    return warnings_acumulados


class ModeloAjustado(TypedDict):
    model: ARIMA
    warnings: dict[str, int]


def ajustar_arima(time_series: pd.DataFrame, p: int, d: int, q: int) -> ModeloAjustado:
    with warnings.catch_warnings(record=True) as warns:
        model = ARIMA(time_series, order=(p, d, q))
        model_fit = model.fit()
    tipos_warnings = count_warnings(warns)
    return ModeloAjustado(model=model_fit, warnings=tipos_warnings)


class MetricasModelo(TypedDict):
    # pep8: nomes de variaveis são snake case (lower com underline separando)
    AIC: float
    BIC: float
    RMSE: float
    # yhat: pd.Series


def obter_metricas_modelo(modelo, time_series):
    predictions = modelo.forecast(steps=len(time_series))
    rmse = root_mean_squared_error(time_series, predictions)
    # yhat = modelo.predict()
    return MetricasModelo(RMSE=rmse, AIC=modelo.aic, BIC=modelo.bic)


def cross_validate_arimas(p_values, d_values, q_values, time_series, split):
    resultados_resumido = {}
    params = {}

    # Considerar fazer isso fora da funcao e receber direto os params.
    for p, d, q in product(p_values, d_values, q_values):
        params[f"ARIMA({p=}, {d=}, {q=})"] = {"p": p, "d": d, "q": q}

    for name, param in params.items():
        resultados_modelo = []
        for train_index, test_index in split.split(time_series):
            train, test = time_series.iloc[train_index], time_series.iloc[test_index]

            try:
                ajuste_atual = ajustar_arima(train, param["p"], param["d"], param["q"])
            except Exception as e:
                print(
                    f"Erro ao ajustar o modelo ARIMA(p={param['p']}, d={param['d']}, {param['q']}): {e}"
                )
                continue
            metricas_atual = obter_metricas_modelo(ajuste_atual["model"], test)

            resultados_modelo.append(metricas_atual)
        tamanho = len(resultados_modelo)
        resultados_resumido[name] = {
            "p": param["p"],
            "d": param["d"],
            "q": param["q"],
            "AIC": sum([metricas["AIC"] for metricas in resultados_modelo]) / tamanho,
            "BIC": sum([metricas["BIC"] for metricas in resultados_modelo]) / tamanho,
            "RMSE": sum([metricas["RMSE"] for metricas in resultados_modelo]) / tamanho,
        }

    return resultados_resumido


def computar_melhores_modelos(resultados_consolidados):
    modelos_aic = []
    modelos_bic = []
    modelos_rmse = []
    total_warnings_aic = Counter()
    total_warnings_bic = Counter()
    total_warnings_rmse = Counter()

    for resultado in resultados_consolidados:
        modelos_aic.append(resultado["Menor AIC"]["Modelo"])
        modelos_bic.append(resultado["Menor BIC"]["Modelo"])
        modelos_rmse.append(resultado["Menor RMSE"]["Modelo"])

        for warning, count in resultado["Menor AIC"]["Warnings"].items():
            total_warnings_aic[warning] += count
        for warning, count in resultado["Menor BIC"]["Warnings"].items():
            total_warnings_bic[warning] += count
        for warning, count in resultado["Menor RMSE"]["Warnings"].items():
            total_warnings_rmse[warning] += count

    moda_aic = Counter(modelos_aic).most_common(1)[0][0]
    moda_bic = Counter(modelos_bic).most_common(1)[0][0]
    moda_rmse = Counter(modelos_rmse).most_common(1)[0][0]

    print(
        f"Moda do modelo de melhor AIC: {moda_aic}. Total de Warnings: {dict(total_warnings_aic)}"
    )
    print(
        f"Moda do modelo de melhor BIC: {moda_bic}. Total de Warnings: {dict(total_warnings_bic)}"
    )
    print(
        f"Moda do modelo de melhor RMSE: {moda_rmse}. Total de Warnings: {dict(total_warnings_rmse)}"
    )

    melhores_modelos = {"AIC": moda_aic, "BIC": moda_bic, "RMSE": moda_rmse}

    return melhores_modelos


def gera_graficos_predict(df, coluna_serie, yhat, p, d, q):
    # alterar para calcular o yhat aqui
    plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    grafico_yhat = sns.lineplot(x=df.index, y=df[coluna_serie], label="Real")
    sns.lineplot(x=df.index, y=yhat, color="red", label="Previsto")
    grafico_yhat.set_title(f"Gráfico para ARIMA({p}, {d}, {q})")
    grafico_yhat.set_xlabel("Eixo X")
    grafico_yhat.set_ylabel("Eixo Y")


def gera_ljungbox(model_fit, p, d, q):
    ljungbox = model_fit.test_serial_correlation(method="ljungbox")

    print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, {d}, {q}):\n")

    for lag in range(len(ljungbox[0][0])):
        p_valor = ljungbox[0][1][lag]
        print(f"Lag {lag + 1}: p-valor = {p_valor}")


def gera_diagnosticos(model_fit, p, d, q):
    tela_diagnostics = plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    model_fit.plot_diagnostics(fig=tela_diagnostics)

    tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p=}, {d=}, {q=})")
