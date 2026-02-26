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
    agent_steps = []

    # ---------------- Agent reasoning ----------------
    agent_steps.append("🔍 Reading and understanding dataset structure")
    insights.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    agent_steps.append("🧪 Checking for missing values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        insights.append("No missing values were found in the dataset.")
    else:
        insights.append("Some columns contain missing values.")

    # ---------------- Dataset quality ----------------
    agent_steps.append("📊 Evaluating dataset quality")
    score = 100
    score -= int((missing.sum() / (df.shape[0] * df.shape[1])) * 100)
    score -= min(20, df.duplicated().sum())
    score = max(score, 0)
    insights.insert(0, f"📊 Dataset Quality Score: {score} / 100")

    # ---------------- Feature analysis ----------------
    agent_steps.append("📊 Identifying feature types")

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns
    datetime_cols = df.select_dtypes(include="datetime").columns

    agent_steps.append("📈 Generating adaptive visualizations")

    # Clear old plots
    for f in os.listdir(PLOT_FOLDER):
        os.remove(os.path.join(PLOT_FOLDER, f))

    # Numeric features
    for col in numeric_cols:
        plt.figure()
        unique_vals = df[col].nunique()

        if unique_vals <= 2:
            sns.countplot(x=df[col])
            insights.append(f"Binary feature '{col}' visualized using a count plot.")
        else:
            sns.histplot(df[col].dropna(), kde=True)
            insights.append(f"Numeric feature '{col}' visualized using a histogram.")

        plt.title(col)
        plt.savefig(f"{PLOT_FOLDER}/{col}.png")
        plt.close()

    # Categorical features
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            plt.figure()
            df[col].value_counts().plot(kind="bar")
            plt.title(col)
            plt.savefig(f"{PLOT_FOLDER}/{col}.png")
            plt.close()
            insights.append(f"Categorical feature '{col}' visualized using a bar chart.")

    # Correlation
    if len(numeric_cols) > 1:
        agent_steps.append("🔗 Performing correlation analysis")
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(f"{PLOT_FOLDER}/correlation.png")
        plt.close()
        insights.append("Correlation analysis was performed on numerical features.")

    agent_steps.append("✅ Automated EDA completed")

    return insights, agent_steps


@app.route("/", methods=["GET", "POST"])
def index():
    insights = []
    agent_steps = ["⏳ Agent waiting for dataset upload..."]
    plots = []

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            df = pd.read_csv(path)

            insights, agent_steps = automated_eda_agent(df)
            plots = os.listdir(PLOT_FOLDER)

    return render_template(
        "index.html",
        insights=insights,
        agent_steps=agent_steps,
        plots=plots
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
