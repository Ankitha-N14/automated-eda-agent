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
    activity_log = []

    # Step 1: Dataset overview
    activity_log.append("🔍 Reading and understanding dataset structure")
    insights.append(
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    # Step 2: Missing values
    activity_log.append("🧪 Checking for missing values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append("Some columns contain missing values.")
    else:
        insights.append("No missing values were found in the dataset.")

    # Step 3: Dataset Quality Score
    activity_log.append("📊 Evaluating dataset quality")
    score = 100
    missing_ratio = missing.sum() / (df.shape[0] * df.shape[1])
    score -= int(missing_ratio * 100)

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        score -= min(20, duplicates)

    score = max(score, 0)
    insights.insert(0, f"📊 Dataset Quality Score: {score} / 100")

    # Step 4: Identify numerical columns
    activity_log.append("📊 Identifying numerical features")
    numeric_cols = df.select_dtypes(include="number").columns

    # Step 5: Generate distribution plots
    activity_log.append("📈 Generating distribution plots")
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plot_path = f"{PLOT_FOLDER}/{col}_hist.png"
        plt.savefig(plot_path)
        plt.close()
        insights.append(f"Column '{col}' shows its distribution as plotted.")

    # Step 6: Correlation analysis
    if len(numeric_cols) > 1:
        activity_log.append("🔗 Performing correlation analysis")
        plt.figure(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        corr_path = f"{PLOT_FOLDER}/correlation.png"
        plt.savefig(corr_path)
        plt.close()
        insights.append(
            "Correlation analysis was performed on numerical features."
        )

    # Step 7: Completion
    activity_log.append("✅ Automated EDA completed")

    return insights, activity_log


# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    insights = []
    activity_log = []
    plots = []

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            insights, activity_log = automated_eda_agent(df)
            plots = os.listdir(PLOT_FOLDER)

    return render_template(
        "index.html",
        insights=insights,
        activity_log=activity_log,
        plots=plots
    )


# ------------------------
# Render-compatible runner
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
