import joblib
import numpy as np
from flask import Flask
from flask import redirect
from flask import render_template
from flask import session
from flask import url_for
from flask_wtf import FlaskForm
from tensorflow.keras.models import load_model
from wtforms import SubmitField
from wtforms import TextField


def return_prediction(model, scaler, sample_json):

    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]

    flower = [[s_len, s_wid, p_len, p_wid]]

    flower = scaler.transform(flower)

    classes = np.array(["setosa", "versicolor", "virginica"])

    class_ind = model.predict_classes(flower)

    return classes[class_ind][0]


app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"


class FlowerForm(FlaskForm):

    sep_len = TextField("Sepal Length")
    sep_wid = TextField("Sepal Width")
    pet_len = TextField("Petal Length")
    pet_wid = TextField("Petal Width")

    submit = SubmitField("Analyze")


@app.route("/", methods=["GET", "POST"])
def index():

    form = FlowerForm()

    if form.validate_on_submit():

        session["sep_len"] = form.sep_len.data
        session["sep_wid"] = form.sep_wid.data
        session["pet_len"] = form.pet_len.data
        session["pet_wid"] = form.pet_wid.data

        return redirect(url_for("prediction"))

    return render_template("home.html", form=form)


flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")


@app.route("/prediction")
def prediction():

    content = {}

    content["sepal_length"] = float(session["sep_len"])
    content["sepal_width"] = float(session["sep_wid"])
    content["petal_length"] = float(session["pet_len"])
    content["petal_width"] = float(session["pet_wid"])

    results = return_prediction(flower_model, flower_scaler, content)

    return render_template("prediction.html", results=results)


if __name__ == "__main__":
    app.run()
