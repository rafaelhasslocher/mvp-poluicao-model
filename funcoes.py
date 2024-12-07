import warnings
from itertools import product
from typing import TypedDict

import matplotlib.pyplot as plt
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

def obter_metricas_modelo(modelo, time_series):
    predictions = modelo.predict(start=time_series.index[0], end=time_series.index[-1])
    rmse = root_mean_squared_error(time_series, predictions)
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

    resultados_resumido = pd.DataFrame.from_dict(resultados_resumido, orient='index')

    print(f"O modelo de menor AIC médio é o {resultados_resumido['AIC'].idxmin()}.")
    print(f"O modelo de menor BIC médio é o {resultados_resumido['BIC'].idxmin()}.")
    print(f"O modelo de menor RMSE é o {resultados_resumido['RMSE'].idxmin()}.")

    return resultados_resumido


def gera_graficos_predict(time_series, p, d, q):
    modelo = ajustar_arima(time_series, p, d, q)
    yhat = modelo["model"].predict(start=time_series.index[0], end=time_series.index[-1])
    plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    grafico_yhat = sns.lineplot(x=time_series.index, y=time_series, label="Real")
    sns.lineplot(x=time_series.index, y=yhat, color="red", label="Previsto")
    grafico_yhat.set_title(f"Gráfico para ARIMA({p}, {d}, {q})")
    grafico_yhat.set_xlabel("Eixo X")
    grafico_yhat.set_ylabel("Eixo Y")


def gera_ljungbox(time_series, p, d, q):

    ajuste = ajustar_arima(time_series, p, d, q)
    model = ajuste["model"]

    ljungbox = model.test_serial_correlation(method="ljungbox")

    print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, {d}, {q}):\n")

    for lag in range(len(ljungbox[0][0])):
        p_valor = ljungbox[0][1][lag]
        print(f"Lag {lag + 1}: p-valor = {p_valor}")


def gera_diagnosticos(time_series, p, d, q):

    ajuste = ajustar_arima(time_series, p, d, q)
    model = ajuste["model"]

    tela_diagnostics = plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    model.plot_diagnostics(fig=tela_diagnostics, auto_ylims=True)

    tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p=}, {d=}, {q=})")
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

