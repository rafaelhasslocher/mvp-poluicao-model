import warnings
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from collections import Counter


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


def ajustar_arimas(coluna_serie, train, test, name, param):
    try:
        resultados_iteracao = {}
        with warnings.catch_warnings(record=True) as warns:

            p = param["p"]
            d = param["d"]
            q = param["q"]

            model = ARIMA(
                train[coluna_serie], order=(p, d, q)
            )
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            rmse = root_mean_squared_error(test[coluna_serie], predictions)
            tipos_warnings = count_warnings(warns)
            yhat = model_fit.predict(start=0, end=len(train[coluna_serie]) - 1)
        resultados_iteracao[name] = {
            **param,
            "Modelo": model_fit,
            "warnings": tipos_warnings,
            "AIC": model_fit.aic,
            "BIC": model_fit.bic,
            "RMSE": rmse,
            "yhat": yhat,
        }

    except Exception as e:
        print(
            f"Erro ao ajustar o modelo ARIMA(p={param['p']}, d={param['d']}, {param['q']}): {e}"
        )
    return resultados_iteracao, tipos_warnings


def cross_validate_arimas(p_values, d_values, q_values, df, split, coluna_serie):
    resultados = []
    resultados_detalhados = []
    params = {}
    warnings_acumulados_aic = {}
    warnings_acumulados_bic = {}
    warnings_acumulados_rmse = {}

    for p, d, q in product(p_values, d_values, q_values):
        params[f"ARIMA({p=}, {d=}, {q=})"] = {"p": p, "d": d, "q": q}

    for train_index, test_index in split.split(df):
        resultados_iteracao = {}
        menor_aic = np.inf
        menor_bic = np.inf
        menor_rmse = np.inf
        melhor_modelo_aic = None
        melhor_modelo_bic = None
        melhor_modelo_rmse = None

        for name, param in params.items():
            train, test = df.iloc[train_index], df.iloc[test_index]

            resultados_iteracao, tipos_warnings = ajustar_arimas(
                coluna_serie, train, test, name, param
            )

            for modelo, detalhes in resultados_iteracao.items():
                if detalhes["AIC"] < menor_aic:
                    menor_aic = detalhes["AIC"]
                    melhor_modelo_aic = modelo

                if detalhes["BIC"] < menor_bic:
                    menor_bic = detalhes["BIC"]
                    melhor_modelo_bic = modelo

                if detalhes["RMSE"] < menor_rmse:
                    menor_rmse = detalhes["RMSE"]
                    melhor_modelo_rmse = modelo

            if name == melhor_modelo_aic:
                warnings_acumulados_aic = acumula_warnings(tipos_warnings)

            if name == melhor_modelo_bic:
                warnings_acumulados_bic = acumula_warnings(tipos_warnings)

            if name == melhor_modelo_rmse:
                warnings_acumulados_rmse = acumula_warnings(tipos_warnings)

            consolidacao = {
                "Menor AIC": {
                    "Modelo": melhor_modelo_aic,
                    "AIC": menor_aic,
                    "Warnings": warnings_acumulados_aic,
                },
                "Menor BIC": {
                    "Modelo": melhor_modelo_bic,
                    "BIC": menor_bic,
                    "Warnings": warnings_acumulados_bic,
                },
                "Menor RMSE": {
                    "Modelo": melhor_modelo_rmse,
                    "RMSE": menor_rmse,
                    "Warnings": warnings_acumulados_rmse,
                },
            }

            resultados.append(consolidacao)
            resultados_detalhados.append(resultados_iteracao)

    return resultados, resultados_detalhados


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

    print(f"Moda do modelo de melhor AIC: {moda_aic}. Total de Warnings: {dict(total_warnings_aic)}")
    print(f"Moda do modelo de melhor BIC: {moda_bic}. Total de Warnings: {dict(total_warnings_bic)}")
    print(f"Moda do modelo de melhor RMSE: {moda_rmse}. Total de Warnings: {dict(total_warnings_rmse)}")

def gera_graficos_predict(df, coluna_serie, yhat, p, d, q):
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

