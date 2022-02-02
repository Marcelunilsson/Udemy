# %%
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential


# %%
df = pd.read_csv(
    "/home/marcel/Udemy/Py_DS_ML_Bootcamp/22-Deep Learning/DATA/cancer_classification.csv"
)
# %%
df.info()
# %%
df.describe().transpose()
# %%
sns.countplot(x="benign_0__mal_1", data=df)
# %%
df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
# %%
sns.heatmap(df.corr(), annot=False, cmap="coolwarm_r")
# %%
X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values
# %%
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# %%
# %%
scaler = MinMaxScaler()
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%

# %%
X_train.shape
# %%
model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))


model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")
# %%
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))
# %%
losses = pd.DataFrame(model.history.history)
# %%
losses.plot()
# %%
model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))


model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")
# %%
# %%
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
# %%
model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
)
# %%
losses = pd.DataFrame(model.history.history)
# %%
losses.plot()

# %%
# %%
model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(15, activation="relu"))
model.add(Dropout(0.3))


model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")
# %%
# %%
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
# %%
model.fit(
    x=X_train,
    y=y_train,
    epochs=600,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
)
# %%
losses = pd.DataFrame(model.history.history)
# %%
losses.plot()

# %%
y_pred = model.predict_classes(X_test)
# %%
# %%
cmd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
# %%
cmd.plot()
# %%
print(classification_report(y_test, y_pred))
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
