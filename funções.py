import warnings
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


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




def ajustar_arimas(p_values, d_values, q_values, df, split, coluna_serie):
    resultados_consolidados = {}
    params = {}
    resultados = {}

    for p, d, q in product(p_values, d_values, q_values):
        params[f"ARIMA({p=}, {d=}, {q=})"] = {"p": p, "d": d, "q": q}

    for i, (train_index, test_index) in enumerate(split.split(df)):
        resultados_iteracao = {}
        menor_aic = np.inf
        menor_bic = np.inf
        menor_rmse = np.inf
        melhor_modelo_aic = None
        melhor_modelo_bic = None
        melhor_modelo_rmse = None

        for name, param in params.items():
            train, test = df.iloc[train_index], df.iloc[test_index]
            try:
                with warnings.catch_warnings(record=True) as warns:
                    model = ARIMA(
                        train[coluna_serie], order=(param["p"], param["d"], param["q"])
                    )
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=len(test))
                    rmse = root_mean_squared_error(test[coluna_serie], predictions)
                resultados_iteracao[name] = {
                    **param,
                    "Modelo": model_fit,
                    "warnings": [warn.message for warn in warns],
                    "AIC": model_fit.aic,
                    "BIC": model_fit.bic,
                    "RMSE": rmse,
                }
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

            except Exception as e:
                print(
                    f"Erro ao ajustar o modelo ARIMA(p={param['p']}, d={param['d']}, {param['q']}): {e}"
                )
        resultados[i] = resultados_iteracao
        resultados_consolidados[i] = {
            "Iteração": i,
            "Menor AIC": {"Modelo": melhor_modelo_aic, "AIC": menor_aic},
            "Menor BIC": {"Modelo": melhor_modelo_bic, "BIC": menor_bic},
            "Menor RMSE": {"Modelo": melhor_modelo_rmse, "RMSE": rmse},
        }

    return resultados, resultados_consolidados



# Exemplo de uso
# p_values = [1, 2, 3]
# d_values = [0, 1]
# q_values = [1, 2, 3]
# df = pd.DataFrame({'value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]})
# split = TimeSeriesSplit(n_splits=3)
# coluna_serie = 'value'
# resultados, resultados_consolidados = ajustar_arimas(p_values, d_values, q_values, df, split, coluna_serie)


# Exemplo de uso
# p_values = [1, 2, 3]
# d_values = [0, 1]
# q_values = [1, 2, 3]
# df = pd.DataFrame({'value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]})
# split = TimeSeriesSplit(n_splits=3)
# coluna_serie = 'value'
# resultados_consolidados = ajustar_arimas(p_values, d_values, q_values, df, split, coluna_serie)


def encontrar_melhores_modelos(resultados):
    melhores_modelos = {}

    for indice, modelos in resultados.items():
        melhor_aic = np.inf
        melhor_bic = np.inf
        melhor_modelo_aic = None
        melhor_modelo_bic = None

        for nome_modelo, detalhes in modelos.items():
            if detalhes["AIC"] < melhor_aic:
                melhor_aic = detalhes["AIC"]
                melhor_modelo_aic = nome_modelo

            if detalhes["BIC"] < melhor_bic:
                melhor_bic = detalhes["BIC"]
                melhor_modelo_bic = nome_modelo

        melhores_modelos[indice] = {
            "Melhor AIC": {"Modelo": melhor_modelo_aic, "AIC": melhor_aic},
            "Melhor BIC": {"Modelo": melhor_modelo_bic, "BIC": melhor_bic},
        }

    return melhores_modelos


# Exemplo de uso
resultados = {
    0: {
        "ARIMA(p=1, d=0, q=1)": {
            "p": 1,
            "d": 0,
            "q": 1,
            "AIC": 7221.244,
            "BIC": 7238.891,
            "RMSE": 110.641,
        },
        "ARIMA(p=1, d=0, q=2)": {
            "p": 1,
            "d": 0,
            "q": 2,
            "AIC": 7220.700,
            "BIC": 7242.759,
            "RMSE": 110.630,
        },
        # Outros modelos...
    },
    1: {
        "ARIMA(p=1, d=0, q=1)": {
            "p": 1,
            "d": 0,
            "q": 1,
            "AIC": 14584.228,
            "BIC": 14604.645,
            "RMSE": 98.194,
        },
        "ARIMA(p=1, d=0, q=2)": {
            "p": 1,
            "d": 0,
            "q": 2,
            "AIC": 14581.246,
            "BIC": 14606.767,
            "RMSE": 98.206,
        },
        # Outros modelos...
    },
}

melhores_modelos = encontrar_melhores_modelos(resultados)
print(melhores_modelos)


# acuracia = []

# for nome_modelo, modelo in resultados.items():
#     p = modelo["p"]
#     d = modelo["d"]
#     q = modelo["q"]
#     model_fit = modelo["Modelo"]
#     aic = modelo["AIC"]
#     bic = modelo["BIC"]
#     acuracia = modelo["Acurácia"]

#     yhat = model_fit.predict(start=0, end=len(df[coluna_serie]) - 1)

#     predict = gera_graficos_predict(df, coluna_serie, yhat, p, d, q)

#     ljungbox = gera_ljungbox(model_fit, p, d, q)

#     diagnosticos = gera_diagnosticos(model_fit, p, d, q)

# df_resultados = pd.DataFrame(resultados.values())

# max_aic = df_resultados.loc[df_resultados["AIC"].idxmax()]

# Exemplo de uso
# p_values = [1, 2, 3]
# d_values = [0, 1]
# q_values = [1, 2, 3]
# df = pd.DataFrame({'value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]})
# split = TimeSeriesSplit(n_splits=3)
# coluna_serie = 'value'
# resultados = ajustar_arimas(p_values, d_values, q_values, df, split, coluna_serie)


# def testar_modelos(p, d, q, df, split):
#     melhores_modelos = []
#     melhor_aic = None
#     for train, test = df,


# best_models = []

# for train_index, test_index in tscv.split(data):
#     train, test = data.iloc[train_index], data.iloc[test_index]

#     best_aic = np.inf
#     best_order = None
#     best_model_fit = None

#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 try:
#                     model_fit = fit_arima_model(train["value"], order=(p, d, q))
#                     aic = model_fit.aic
#                     if aic < best_aic:
#                         best_aic = aic
#                         best_order = (p, d, q)
#                         best_model_fit = model_fit
#                 except Exception as e:
#                     print(f"Erro ao ajustar o modelo ARIMA(p={p}, d={d}, q={q}): {e}")

#     best_models.append((best_order, best_model_fit))

# # Comparar os melhores modelos selecionados
# for i, (order, model_fit) in enumerate(best_models):
#     print(f"Divisão {i}: Melhor modelo ARIMA{order} com AIC = {model_fit.aic}")
