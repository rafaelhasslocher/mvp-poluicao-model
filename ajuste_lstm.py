import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model  # type: ignore
from keras.layers import Dense, LSTM, Dropout, Input  # type: ignore
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore
from tensorflow.keras.optimizers import SGD, Adam, RMSprop  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from keras_tuner import RandomSearch

from ajuste_arima import (
    coluna_serie,
    base_treino,
    base_teste,
    residuos_treino,
    residuos_teste,
)

base_treino = base_treino.drop(columns={coluna_serie})

base_treino["residuos"] = residuos_treino

base_treino = base_treino[
    ["residuos", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
]

base_teste = base_teste.drop(columns={coluna_serie})

base_teste["residuos"] = residuos_teste

df_test = base_teste[
    ["residuos", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
]

base_teste

base_treino.describe()

df_train_scaled = base_treino.copy()
df_test_scaled = df_test.copy()


mapping = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}


df_train_scaled["wnd_dir"] = df_train_scaled["wnd_dir"].map(mapping)
df_test_scaled["wnd_dir"] = df_test_scaled["wnd_dir"].map(mapping)


values = df_train_scaled.values


# groups = [1, 2, 3, 5, 6]
# i = 1


# plt.figure(figsize=(20, 14))
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group], color=cm.plasma(group / len(groups)))
#     plt.xlabel("Index")
#     plt.title(base_treino.columns[group], y=0.75, loc="right", fontsize=15)
#     i += 1
# plt.show()

# sns.set_theme(style="darkgrid")

# fig, axs = plt.subplots(3, 2, figsize=(24, 14))

# sns.histplot(
#     data=df_test_scaled, x="residuals", kde=True, color="skyblue", ax=axs[0, 0]
# )
# sns.histplot(data=df_test_scaled, x="dew", kde=True, color="olive", ax=axs[0, 1])
# sns.histplot(data=df_test_scaled, x="temp", kde=True, color="gold", ax=axs[1, 0])
# sns.histplot(data=df_test_scaled, x="press", kde=True, color="teal", ax=axs[1, 1])
# sns.histplot(
#     data=df_test_scaled, x="wnd_dir", kde=True, color="steelblue", ax=axs[2, 0]
# )
# sns.histplot(
#     data=df_test_scaled, x="wnd_spd", kde=True, color="goldenrod", ax=axs[2, 1]
# )

# plt.show()

scaler = MinMaxScaler()


columns = ["residuos", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]


df_train_scaled[columns] = scaler.fit_transform(df_train_scaled[columns])
df_test_scaled[columns] = scaler.transform(df_test_scaled[columns])


df_train_scaled = np.array(df_train_scaled)
df_test_scaled = np.array(df_test_scaled)

X = []
y = []
n_future = 1
n_past = 11


for i in range(n_past, len(df_train_scaled) - n_future + 1):
    X.append(df_train_scaled[i - n_past : i, 1 : df_train_scaled.shape[1]])
    y.append(df_train_scaled[i + n_future - 1 : i + n_future, 0])
X_train, y_train = np.array(X), np.array(y)


X = []
y = []
for i in range(n_past, len(df_test_scaled) - n_future + 1):
    X.append(df_test_scaled[i - n_past : i, 1 : df_test_scaled.shape[1]])
    y.append(df_test_scaled[i + n_future - 1 : i + n_future, 0])
X_test, y_test = np.array(X), np.array(y)

print(
    "X_train shape : {}   y_train shape : {} \n"
    "X_test shape : {}      y_test shape : {} ".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )
)

def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units_1", min_value=32, max_value=512, step=32),
            return_sequences=True,
        )
    )
    model.add(Dropout(hp.Float("dropout_1", min_value=0.05, max_value=0.5, step=0.05)))
    model.add(
        LSTM(
            units=hp.Int("units_2", min_value=16, max_value=128, step=16),
            return_sequences=False,
        )
    )
    model.add(Dense(y_train.shape[1]))

    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.1, sampling="log"))
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.1, sampling="log"))
    else:
        optimizer = RMSprop(learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.1, sampling="log"))

    loss_choice = hp.Choice('loss', ['mse', 'mae', 'huber'])

    model.compile(
        loss=loss_choice,
        optimizer=optimizer,
        metrics=[RootMeanSquaredError()],
    )
    return model



early_stopping = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True
)

tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=5,
    executions_per_trial=1,
    directory="my_dir",
    project_name="lstm_tuning",
)

tuner.search(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    shuffle=False,
)

# Obter os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters()[0]

# Visualizar os valores dos melhores hiperparâmetros
print(f"Melhor número de unidades na primeira camada LSTM: {best_hps.get('units_1')}")
print(f"Melhor taxa de dropout na primeira camada: {best_hps.get('dropout_1')}")
print(f"Melhor número de unidades na segunda camada LSTM: {best_hps.get('units_2')}")
print(f"Melhor taxa de aprendizado: {best_hps.get('learning_rate')}")
print(f"Melhor otimizador: {best_hps.get('optimizer')}")
print(f"Melhor função de perda: {best_hps.get('loss')}")


model = build_model(best_hps)
history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    shuffle=False,
)


best_model = load_model("best_model.keras")


test_predictions = best_model.predict(X_test)

vitor = np.repeat(test_predictions, 8, axis=1)

vitor = scaler.inverse_transform(vitor)

vitor = vitor[:, 0]

y_test_vitor = np.repeat(y_test, 8, axis=1)

y_test_vitor = scaler.inverse_transform(y_test_vitor)

y_test_vitor = y_test_vitor[:, 0]

test_results = pd.DataFrame(data={"Train Predictions": vitor, "Actual": y_test_vitor})
test_results.head()

plt.plot(test_results["Train Predictions"][:200], label="Predicted Values")
plt.plot(test_results["Actual"][:200], label="True Values")
plt.legend()
rmse = sqrt(mse(y_test, test_predictions))
print("Test RMSE: %.5f" % rmse)
