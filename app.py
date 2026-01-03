from flask import Flask, render_template, request
import joblib
import numpy as np
import google.generativeai as genai

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

genai.configure(api_key="AIzaSyAOWEYpaTpsYbT6GXBx1HPH5WB_ltn7gP0")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

labels = {
    1: "Not Addicted",
    2: "Low Addiction",
    3: "Moderate Addiction",
    4: "High Addiction",
    5: "Severe Addiction"
}

features = [
    "Age","Gender","Avg_Social_Media_Usage","Dominant_Platform",
    "Checking_Frequency","Pre_Sleep_Usage","Content_Type",
    "Sleep_Latency","Total_Sleep_Time","Sleep_Efficiency",
    "Sleep_Quality","Wake_After_Sleep_Onset","Awakenings",
    "Ease_of_Sleep","Stress_Cortisol","Blue_Light",
    "Stress_Level","Anxiety_Depression","Restlessness",
    "Interest_Fluctuation","Relationship_Status","Loneliness"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_values = [float(request.form[f]) for f in features]
    input_array = np.array(input_values).reshape(1, -1)

    prediction = int(model.predict(input_array)[0])
    label = labels[prediction]

    prompt = f"""Analyze the following user behavior data and provide personalized recommendations:
User Inputs: {dict(zip(features, input_values))}
Predicted Social Media Addiction Level: {label}

Provide meaningful recommendations based on the addiction level.
Provide useful insights that would help the participant in reducing social media addiction.
Provide the insights in one paragraph in a >100 words in a concise way.
make each in one line sentence , dont use * * in title too and anywhere and use numerical points
make personalized suggestion max 4 lines , highlight the key areas to focus on.
"""


    gemini_text = gemini_model.generate_content(prompt).text

    return render_template(
        "result.html",
        prediction=label,
        suggestions=gemini_text
    )

if __name__ == "__main__":
    app.run(debug=True)
