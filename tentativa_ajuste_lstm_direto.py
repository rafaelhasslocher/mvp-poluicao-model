import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras_tuner import RandomSearch

# from ajuste_arima import (
#     coluna_serie,
#     base_treino,
#     base_teste,
#     residuos_treino,
#     residuos_teste,
# )

df_train = pd.read_csv("data/LSTM-Multivariate_pollution.csv")
df_train

df_train.info()

df_test = pd.read_csv("data/pollution_test_data1.csv")
df_test

print(df_train.isnull().sum(), "\n---------------- \n", df_test.isnull().sum())

df_train.describe()

df_train_scaled = df_train.copy()
df_test_scaled = df_test.copy()

# Define the mapping dictionary
mapping = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}

# Replace the string values with numerical values
df_train_scaled["wnd_dir"] = df_train_scaled["wnd_dir"].map(mapping)
df_test_scaled["wnd_dir"] = df_test_scaled["wnd_dir"].map(mapping)

df_train_scaled["date"] = pd.to_datetime(df_train_scaled["date"])
# Resetting the index
df_train_scaled.set_index("date", inplace=True)
df_train_scaled.head()

values = df_train_scaled.values

# specify columns to plot
groups = [1, 2, 3, 5, 6]
i = 1

# plot each column
plt.figure(figsize=(20, 14))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group], color=cm.plasma(group / len(groups)))
    plt.xlabel("Index")
    plt.title(df_train.columns[group], y=0.75, loc="right", fontsize=15)
    i += 1
plt.show()

sns.set(style="darkgrid")

fig, axs = plt.subplots(3, 2, figsize=(24, 14))

sns.histplot(
    data=df_test_scaled, x="pollution", kde=True, color="skyblue", ax=axs[0, 0]
)
sns.histplot(data=df_test_scaled, x="dew", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df_test_scaled, x="temp", kde=True, color="gold", ax=axs[1, 0])
sns.histplot(data=df_test_scaled, x="press", kde=True, color="teal", ax=axs[1, 1])
sns.histplot(
    data=df_test_scaled, x="wnd_dir", kde=True, color="steelblue", ax=axs[2, 0]
)
sns.histplot(
    data=df_test_scaled, x="wnd_spd", kde=True, color="goldenrod", ax=axs[2, 1]
)

plt.show()

scaler = MinMaxScaler()

# Define the columns to scale
columns = ["pollution", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]

df_test_scaled = df_test_scaled[columns]

# Scale the selected columns to the range 0-1
df_train_scaled[columns] = scaler.fit_transform(df_train_scaled[columns])
df_test_scaled[columns] = scaler.transform(df_test_scaled[columns])

# Show the scaled data
df_train_scaled.head()

df_test_scaled.head()

df_train_scaled = np.array(df_train_scaled)
df_test_scaled = np.array(df_test_scaled)

X = []
y = []
n_future = 1
n_past = 11

#  Train Sets
for i in range(n_past, len(df_train_scaled) - n_future + 1):
    X.append(df_train_scaled[i - n_past : i, 1 : df_train_scaled.shape[1]])
    y.append(df_train_scaled[i + n_future - 1 : i + n_future, 0])
X_train, y_train = np.array(X), np.array(y)

#  Test Sets

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
            units=hp.Int("units_1", min_value=32, max_value=64, step=32),
            return_sequences=True,
        )
    )
    model.add(Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.3, step=0.1)))
    model.add(
        LSTM(
            units=hp.Int("units_2", min_value=16, max_value=32, step=16),
            return_sequences=False,
        )
    )

    model.add(Dense(y_train.shape[1]))
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.001),
        metrics=[RootMeanSquaredError()],
    )
    return model

# # design network

# model = Sequential()
# model.add(
#     LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)
# )
# model.add(Dropout(0.2))
# model.add(LSTM(16, return_sequences=False))
# model.add(Dense(y_train.shape[1]))

# # Compile the model
# model.compile(
#     loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()]
# )

# Define callbacks for avoiding overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True
)

tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=3,
    executions_per_trial=1,
    directory="my_dir",
    project_name="lstm_tuning",
)

tuner.search(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
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

model = build_model(best_hps)
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    shuffle=False,
)


best_model = load_model("best_model.keras")

model.summary()

# fit network
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    shuffle=False,
)

# Load the best model
best_model = load_model("best_model.keras")

plt.figure(figsize=(15, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

test_predictions = best_model.predict(X_test).flatten()
test_results = pd.DataFrame(
    data={"Train Predictions": test_predictions, "Actual": y_test.flatten()}
)
test_results.head()

plt.plot(test_results["Train Predictions"][:200], label="Predicted Values")
plt.plot(test_results["Actual"][:200], label="True Values")
plt.legend()
plt.show()
