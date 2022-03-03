# %%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

plt.rcParams["figure.figsize"] = (12, 8)
# %%
iris = pd.read_csv("../DATA/iris.csv")

X = iris.drop("species", axis=1)
y = iris.species

encoder = LabelBinarizer()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# %%
model = Sequential()

model.add(
    Dense(
        units=4,
        activation="relu",
        input_shape=[
            4,
        ],
    )
)
model.add(Dense(units=3, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# %%
early_stop = EarlyStopping(patience=10)

model.fit(
    x=scaled_X_train, y=y_train, epochs=1200, validation_data=(scaled_X_test, y_test)
)
# %%
metrics = pd.DataFrame(model.history.history)

metrics[["loss", "val_loss"]].plot()
metrics[["accuracy", "val_accuracy"]].plot()
model.evaluate(scaled_X_test, y_test, verbose=0)
# %%

scaled_X = scaler.fit_transform(X)
model = Sequential()

model.add(
    Dense(
        units=4,
        activation="relu",
        input_shape=[
            4,
        ],
    )
)
model.add(Dense(units=3, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x=scaled_X, y=y, epochs=1200)

# %%
# Save model and scaler
model.save("../09-Deployment/final_iris_model.h5")
joblib.dump(scaler, "../09-Deployment/iris_scaler.pkl")

# %%
# Loading models and predicting

flower_model = load_model("../09-Deployment/final_iris_model.h5")
flower_scaler = joblib.load("../09-Deployment/iris_scaler.pkl")
# %%
flower_example = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


# %%
def return_prediction(model, scaler, sample_json):

    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_length"]

    classes = np.array(["setosa", "versicolor", "virginica"])

    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    class_ind = model.predict_classes(flower)[0]

    return classes[class_ind]


# %%
return_prediction(flower_model, flower_scaler, flower_example)
# %%
