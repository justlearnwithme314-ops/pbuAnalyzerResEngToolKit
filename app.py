from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

import subprocess
import os
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_analysis():

    mode = request.form["mode"]

    file = request.files["file"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # --------------------------
    # PREPROCESS ONLY
    # --------------------------

    if mode == "preprocess":

        subprocess.run([
            "python",
            "preproc.py",
            filepath
        ])

        return (
            "<h2>Preprocessing complete.</h2>"
            "<p>processed.csv created.</p>"
            '<a href="/">Back</a>'
        )

    # --------------------------
    # PREPROCESS + ANALYZE
    # --------------------------

    elif mode == "full":

        subprocess.run([
            "python",
            "preproc.py",
            filepath
        ])

        subprocess.run([
            "python",
            "analyze.py",
            "processed.csv"
        ])

        subprocess.run([
            "python",
            "sec2.py",
            "processed.csv",
            "result.csv",
            "reservoir_predictions.csv"
        ])

        return redirect(url_for("results"))

    # --------------------------
    # ANALYZE EXISTING PROCESSED
    # --------------------------

    elif mode == "analyze":

        # overwrite processed.csv
        os.replace(
            filepath,
            "processed.csv"
        )

        subprocess.run([
            "python",
            "analyze.py",
            "processed.csv"
        ])

        subprocess.run([
            "python",
            "sec2.py",
            "processed.csv",
            "result.csv",
            "reservoir_predictions.csv"
        ])

        return redirect(url_for("results"))


@app.route("/results")
def results():

    pbus = pd.read_csv("result.csv")

    preds = pd.read_csv(
        "reservoir_predictions.csv"
    )
    plots = []

    plot_dir = os.path.join(
        "static",
        "plots"
    )

    if os.path.exists(plot_dir):

        plots = sorted(
            os.listdir(plot_dir)
        )

    print("Plots found:", len(plots))

    return render_template(

        "results.html",

        pbus=pbus.to_html(
            classes="table",
            index=True
        ),

        preds=preds.to_html(
            classes="table",
            index=True
        ),

        plots=plots

    )


if __name__ == "__main__":

    app.run(
        debug=True
    )