from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PLOT_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


def automated_eda_agent(df):
    insights = []

    # Basic info
    insights.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append("Some columns contain missing values.")
    else:
        insights.append("No missing values were found in the dataset.")

    # Numeric analysis
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plot_path = f"{PLOT_FOLDER}/{col}_hist.png"
        plt.savefig(plot_path)
        plt.close()
        insights.append(f"Column '{col}' shows its distribution as plotted.")

    # Correlation
    if len(numeric_cols) > 1:
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        corr_path = f"{PLOT_FOLDER}/correlation.png"
        plt.savefig(corr_path)
        plt.close()
        insights.append("Correlation analysis was performed on numerical features.")

    return insights


@app.route("/", methods=["GET", "POST"])
def index():
    insights = []
    plots = []

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            insights = automated_eda_agent(df)

            plots = os.listdir(PLOT_FOLDER)

    return render_template("index.html", insights=insights, plots=plots)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
