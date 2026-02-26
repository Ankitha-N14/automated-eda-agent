from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------
# App configuration
# ------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PLOT_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


# ------------------------
# Automated EDA Agent
# ------------------------
def automated_eda_agent(df):
    insights = []
    agent_steps = []

    # Step 1: Dataset overview
    agent_steps.append("🔍 Reading and understanding dataset structure")
    insights.append(
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    # Step 2: Missing values
    agent_steps.append("🧪 Checking for missing values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append("Some columns contain missing values.")
    else:
        insights.append("No missing values were found in the dataset.")

    # Step 3: Dataset Quality Score
    agent_steps.append("📊 Evaluating dataset quality")
    score = 100
    missing_ratio = missing.sum() / (df.shape[0] * df.shape[1])
    score -= int(missing_ratio * 100)

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        score -= min(20, duplicates)

    score = max(score, 0)
    insights.insert(0, f"📊 Dataset Quality Score: {score} / 100")

    # Step 4: Identify numerical columns
    agent_steps.append("📊 Identifying numerical features")
    numeric_cols = df.select_dtypes(include="number").columns

    # Step 5: Generate distribution plots
    agent_steps.append("📈 Generating distribution plots")
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plot_path = f"{PLOT_FOLDER}/{col}_hist.png"
        plt.savefig(plot_path)
        plt.close()
        insights.append(f"Column '{col}' shows its distribution as plotted.")

    # Step 6: Correlation analysis
    if len(numeric_cols) > 1:
        agent_steps.append("🔗 Performing correlation analysis")
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.savefig(f"{PLOT_FOLDER}/correlation.png")
        plt.close()
        insights.append(
            "Correlation analysis was performed on numerical features."
        )

    # Step 7: Completion
    agent_steps.append("✅ Automated EDA completed")

    return insights, agent_steps


# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # Default state (always visible)
    insights = []
    agent_steps = ["⏳ Agent waiting for dataset upload..."]
    plots = []

    if request.method == "POST":
        # Clear old plots to avoid stale data
        for f in os.listdir(PLOT_FOLDER):
            os.remove(os.path.join(PLOT_FOLDER, f))

        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            insights, agent_steps = automated_eda_agent(df)
            plots = os.listdir(PLOT_FOLDER)

    return render_template(
        "index.html",
        insights=insights,
        agent_steps=agent_steps,
        plots=plots
    )


# ------------------------
# Render-compatible runner
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
