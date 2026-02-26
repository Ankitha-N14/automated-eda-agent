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

    # -------- Agent Reasoning (Level 1) --------
    insights.append("🧠 Agent initialized and waiting for trigger")
    insights.append("🔍 Reading and understanding dataset structure")

    insights.append(
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    insights.append("🧪 Checking for missing values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append("Some columns contain missing values.")
    else:
        insights.append("No missing values were found in the dataset.")

    insights.append("📊 Evaluating dataset quality")
    score = 100
    missing_ratio = missing.sum() / (df.shape[0] * df.shape[1])
    score -= int(missing_ratio * 100)

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        score -= min(20, duplicates)

    score = max(score, 0)
    insights.insert(0, f"📊 Dataset Quality Score: {score} / 100")

    insights.append("📊 Identifying numerical features")
    numeric_cols = df.select_dtypes(include="number").columns

    insights.append("📈 Generating distribution plots")
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.savefig(f"{PLOT_FOLDER}/{col}_hist.png")
        plt.close()
        insights.append(f"Column '{col}' shows its distribution as plotted.")

    if len(numeric_cols) > 1:
        insights.append("🔗 Performing correlation analysis")
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.savefig(f"{PLOT_FOLDER}/correlation.png")
        plt.close()
        insights.append("Correlation analysis was performed on numerical features.")

    insights.append("✅ Automated EDA completed successfully")

    return insights


@app.route("/", methods=["GET", "POST"])
def index():
    insights = []
    plots = []

    if request.method == "POST":
        for f in os.listdir(PLOT_FOLDER):
            os.remove(os.path.join(PLOT_FOLDER, f))

        file = request.files.get("file")
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df = pd.read_csv(path)
            insights = automated_eda_agent(df)
            plots = os.listdir(PLOT_FOLDER)

    return render_template(
        "index.html",
        insights=insights,
        plots=plots
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
