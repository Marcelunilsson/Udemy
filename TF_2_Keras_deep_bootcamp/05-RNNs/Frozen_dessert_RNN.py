# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

plt.rcParams["figure.figsize"] = (12, 8)

# %%

df = pd.read_csv(
    filepath_or_buffer="../DATA/Frozen_Dessert_Production.csv",
    index_col="DATE",
    parse_dates=True,
)
df.columns = ["Production"]

test_size = 24
test_index = len(df) - test_size
train = df.iloc[:test_index]
test = df.iloc[test_index:]

scaler = MinMaxScaler()

train_s = scaler.fit_transform(train)
test_s = scaler.transform(test)

length = 20
batch_size = 1

train_gen = TimeseriesGenerator(
    data=train_s, targets=train_s, length=length, batch_size=batch_size
)

# %%
n_features = 1
model = Sequential()
lstm_units = 256

model.add(
    layer=LSTM(
        units=lstm_units,
        activation="relu",
        # return_sequences=True,
        input_shape=(length, n_features),
    )
)

# model.add(layer=LSTM(units=lstm_units,
#                      activation='relu'))

model.add(layer=Dense(units=1))

model.compile(optimizer="adam", loss="mse")

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=3)

val_gen = TimeseriesGenerator(
    data=test_s, targets=test_s, length=length, batch_size=batch_size
)

model.fit(train_gen, epochs=100, validation_data=val_gen, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
# %%
pred = []

first_eval_batch = train_s[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    pred.append(current_pred)

    current_batch = np.append(
        arr=current_batch[:, 1:, :], values=[[current_pred]], axis=1
    )

true_pred = scaler.inverse_transform(pred)

test["pred"] = true_pred

test.plot()
# %%
RMSE = np.sqrt(mean_squared_error(test.Production, test.pred))

print(f"RMSE: {RMSE}")

# %%

# Forecast

full_scaler = MinMaxScaler()
train_s = full_scaler.fit_transform(df)

length = 24
gen = TimeseriesGenerator(
    data=train_s, targets=train_s, length=length, batch_size=batch_size
)

model = Sequential()

model.add(
    layer=LSTM(
        units=128,
        return_sequences=True,
        activation="relu",
        input_shape=(length, n_features),
    )
)

model.add(layer=LSTM(units=128, activation="relu"))

model.add(layer=Dense(units=1))

model.compile(loss="mse", optimizer="adam")

model.summary()

model.fit(
    gen,
    epochs=10,
)

losses = pd.DataFrame(model.history.history)
losses.plot()
# %%


forecast_length = 48
forecast = []


current_batch = train_s[-length:].reshape((1, length, n_features))

for i in range(forecast_length):

    current_pred = model.predict(current_batch)[0]

    forecast.append(current_pred)

    current_batch = np.append(
        arr=current_batch[:, 1:, :], values=[[current_pred]], axis=1
    )


forecast = scaler.inverse_transform(forecast)
forecast_index = pd.date_range(start="2019-10-01", periods=forecast_length, freq="MS")

forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=["Forecast"])

ax = df.plot()
forecast_df.plot(ax=ax)

# %%
