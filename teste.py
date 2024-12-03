import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller

from funções import (
    ajustar_arimas,
    gera_acf_pacf,
    gera_boxplot,
    # gera_diagnosticos,
    # gera_graficos_predict,
    # gera_ljungbox,
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

plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.mean)
plot_summary_serie(df, coluna_serie, coluna_tempo, periodicidade="ME", summary=np.std)

gera_boxplot(df, coluna_serie, df.index.year)

gera_acf_pacf(df, coluna_serie, lags=10, tipo="acf")
gera_acf_pacf(df, coluna_serie, lags=10, tipo="pacf")

# testando a estacionariedade da série

result = adfuller(df[coluna_serie])
print("Estatística ADF", result[0])
print("p-valor", result[1])

p_values = [1, 2, 3]
q_values = [1, 2, 3]
d_values = [0]


split = TimeSeriesSplit(n_splits=2)
# split = TimeSeriesSplit(n_splits=5)

resultados, resultados_consolidados = ajustar_arimas(
    p_values, d_values, q_values, df, split, coluna_serie
)

resultados
resultados_consolidados

# aic_bic = {}

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
# max_bic = df_resultados.loc[df_resultados["BIC"].idxmax()]
# max_acuracia = df_resultados.loc[df_resultados["Acurácia"].idxmax()]

# print(
#     rf"O modelo de maior AIC foi o ARIMA(p={max_aic['p']}, d={max_aic['d']}, q={max_aic['q']}),"
#     rf" o de maior BIC foi o ARIMA(p={max_bic['p']}, d={max_bic['d']}, q={max_bic['q']})"
#     rf" e o de maior Acurácia foi o ARIMA(p={max_acuracia['p']}, d={max_acuracia['d']}, q={max_acuracia['q']})"
# )


# def gera_analises(df, resultados, coluna_serie):
#     aic_bic = {}
#     tela_yhat, previstos = plt.subplots(
#         len(resultados), 1, figsize=(10, 5 * len(resultados))
#     )
#     tela_yhat.patch.set_facecolor("whitesmoke")
#     for i, (nome_modelo, modelo) in enumerate(resultados.items()):
#         p = modelo['p']
#         q = modelo['q']
#         model_fit = modelo["Modelo"]

#         yhat = model_fit.predict(start=0, end=len(df[coluna_serie]) - 1)

#         sns.lineplot(ax=previstos[i], x=df.index, y=df[coluna_serie], label="Real")
#         sns.lineplot(ax=previstos[i], x=df.index, y=yhat, color="red", label="Previsto")
#         previstos[i].set_title(f"Gráfico para ARIMA({p}, 0, {q})")
#         previstos[i].set_xlabel("Eixo X")
#         previstos[i].set_ylabel("Eixo Y")
#         plt.close(tela_yhat)

#         ljungbox = model_fit.test_serial_correlation(method='ljungbox')

#         mse = mean_squared_error(df[coluna_serie], yhat)
#         rmse = mse ** 0.5
#         print(f"\nAcurácia do modelo ARIMA({p}, 0, {q}): RMSE = {rmse}")

#         print(f"\nResultados do teste de Ljung-Box para ARIMA({p}, 0, {q}):\n")

#         for lag in range(len(ljungbox[0][0])):
#             p_valor = ljungbox[0][1][lag]
#             print(f"Lag {lag + 1}: p-valor = {p_valor}")

#         tela_diagnostics = plt.figure(figsize=(10, 8))
#         tela_diagnostics.patch.set_facecolor("whitesmoke")
#         model_fit.plot_diagnostics(fig=tela_diagnostics)
#         tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p}, 0, {q})")


#         plt.show()
#         plt.close(tela_diagnostics)

#     aic_bic = pd.DataFrame(
#         [
#             (modelo, nome_modelo["AIC"], nome_modelo["BIC"])
#             for modelo, nome_modelo in resultados.items()
#         ],
#         columns=["Modelo", "AIC", "BIC"],
#     )

#     modelo_maximo_aic = aic_bic["Modelo"].loc[aic_bic["AIC"].idxmax()]
#     modelo_maximo_bic = aic_bic["Modelo"].loc[aic_bic["AIC"].idxmax()]

#     p_maximo_aic = resultados[modelo_maximo_aic]["p"]
#     q_maximo_aic = resultados[modelo_maximo_aic]["q"]

#     p_maximo_bic = resultados[modelo_maximo_bic]["p"]
#     q_maximo_bic = resultados[modelo_maximo_bic]["q"]

#     print(
#         rf"O modelo com maior AIC foi o ARIMA({p_maximo_aic}, 0, {q_maximo_aic})"
#         rf" e o de maior BIC foi o ARIMA({p_maximo_bic}, 0, {q_maximo_bic})."
#     )

#     return aic_bic

# aic_bic = gera_analises(df, resultados, coluna_serie)
