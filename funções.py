import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def gera_graficos_predict(df, coluna_serie, yhat, p, d, q):
    plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    grafico_yhat = sns.lineplot(x=df.index, y=df[coluna_serie], label="Real")
    sns.lineplot(x=df.index, y=yhat, color="red", label="Previsto")
    grafico_yhat.set_title(f"Gráfico para ARIMA({p}, 0, {q})")
    grafico_yhat.set_xlabel("Eixo X")
    grafico_yhat.set_ylabel("Eixo Y")


def gera_ljungbox(model_fit, p, d, q):
    ljungbox = model_fit.test_serial_correlation(method="ljungbox")

    print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, 0, {q}):\n")

    for lag in range(len(ljungbox[0][0])):
        p_valor = ljungbox[0][1][lag]
        print(f"Lag {lag + 1}: p-valor = {p_valor}")


def gera_diagnosticos(model_fit, p, d, q):
    tela_diagnostics = plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    model_fit.plot_diagnostics(fig=tela_diagnostics)

    tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p=}, {d=}, {q=})")


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
    plt.figure(figsize=(10, 8), facecolor="whitesmoke")

    if tipo == "acf":
        plot_acf(df[coluna_serie], lags=lags)
        plt.title("Gráfico de Autocorrelação")
        plt.tick_params(labelsize=7)  # noqa: E703

        plt.show()

    else:
        plot_pacf(df[coluna_serie], lags=lags)
        plt.title("Gráfico de Autocorrelação parcial")
        plt.tick_params(labelsize=7)  # noqa: E703

        plt.show()
