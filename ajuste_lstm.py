import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model  # type: ignore
from keras.layers import Dense, LSTM, Dropout  # type: ignore
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from ajuste_arima import (
    residuals_train,
    coluna_serie,
    df_train,
    df_test,
    residuals_test,
)

df_train = df_train.drop(columns={coluna_serie})

df_train["residuals"] = residuals_train

df_train = df_train[
    ["residuals", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
]

df_test = df_test.drop(columns={coluna_serie})

df_test["residuals"] = residuals_test

df_test = df_test[
    ["residuals", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
]

df_test

df_train.describe()

df_train_scaled = df_train.copy()
df_test_scaled = df_test.copy()


mapping = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}


df_train_scaled["wnd_dir"] = df_train_scaled["wnd_dir"].map(mapping)
df_test_scaled["wnd_dir"] = df_test_scaled["wnd_dir"].map(mapping)


values = df_train_scaled.values


groups = [1, 2, 3, 5, 6]
i = 1


plt.figure(figsize=(20, 14))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group], color=cm.plasma(group / len(groups)))
    plt.xlabel("Index")
    plt.title(df_train.columns[group], y=0.75, loc="right", fontsize=15)
    i += 1
plt.show()

sns.set_theme(style="darkgrid")

fig, axs = plt.subplots(3, 2, figsize=(24, 14))

sns.histplot(
    data=df_test_scaled, x="residuals", kde=True, color="skyblue", ax=axs[0, 0]
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


columns = ["dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain", "residuals"]


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

model = Sequential()
model.add(
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)
)
model.add(Dropout(0.2))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(y_train.shape[1]))


model.compile(
    loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint],
    shuffle=False,
)

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

test_predictions_reshaped = test_predictions.reshape(-1, 1)

zeros = np.zeros((test_predictions_reshaped.shape[0], df_test_scaled.shape[1] - 1))

test_predictions_full = np.hstack([zeros, test_predictions_reshaped])
test_predictions_original = scaler.inverse_transform(test_predictions_full)[:, -1]

y_test_reshaped = y_test.flatten().reshape(-1, 1)
zeros_y_test = np.zeros((y_test_reshaped.shape[0], df_test_scaled.shape[1] - 1))
y_test_full = np.hstack([zeros_y_test, y_test_reshaped])
y_test_original = scaler.inverse_transform(y_test_full)[:, -1]

test_results = pd.DataFrame(
    data={"Train Predictions": test_predictions_original}
)
test_results.head()

plt.plot(test_results["Train Predictions"][:200], label="Predicted Values")
plt.legend()
plt.show()

plt.plot(df_test["residuals"][:200], label="Values")
plt.legend()
plt.show()