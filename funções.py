import matplotlib.pyplot as plt
import seaborn as sns

def gera_graficos_predict(df, coluna_serie, yhat, p, d, q):
    tela_yhat = plt.figure(figsize=(10, 8))

    tela_yhat.patch.set_facecolor("whitesmoke")

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
    tela_diagnostics = plt.figure(figsize=(10, 8))
    tela_diagnostics.patch.set_facecolor("whitesmoke")
    model_fit.plot_diagnostics(fig=tela_diagnostics)
    tela_diagnostics.suptitle(f"Gráficos de diagnóstico para ARIMA({p=}, {d=}, {q=})")