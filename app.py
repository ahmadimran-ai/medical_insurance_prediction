import gradio as gr
import pandas as pd
import joblib

# Load trained models
poly = joblib.load("polynomial_features.pkl")
model_poly = joblib.load("insurance_model.pkl")


def predict_insurance(age, bmi, children, gender, smoker, region):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if gender == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    input_poly = poly.transform(input_data)

    prediction = model_poly.predict(input_poly)[0]

    return f"${prediction:,.2f}"


custom_css = """
body {
    background: #0b1220 !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    background: #0b1220 !important;
}

.header {
    text-align: center;
    padding: 30px 10px 20px 10px;
}

.title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

.card {
    background: #111c2e !important;
    border: 1px solid #24344d !important;
    border-radius: 18px !important;
    padding: 25px !important;
}

.result {
    text-align: center;
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #60a5fa !important;
}
"""


with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate"
    )
) as app:

    gr.HTML("""
    <div class="header">
        <div class="title">
            🏥 Medical Insurance Predictor
        </div>
        <div class="subtitle">
            Machine Learning powered insurance cost estimation
        </div>
    </div>
    """)

    with gr.Row():

        with gr.Column(elem_classes="card"):

            gr.Markdown("### 👤 Patient Information")

            age = gr.Slider(
                18, 100,
                value=30,
                step=1,
                label="Age"
            )

            bmi = gr.Slider(
                10, 60,
                value=25,
                step=0.1,
                label="BMI"
            )

            children = gr.Slider(
                0, 10,
                value=0,
                step=1,
                label="Number of Children"
            )

            gender = gr.Radio(
                ["male", "female"],
                value="male",
                label="Gender"
            )

            smoker = gr.Radio(
                ["yes", "no"],
                value="no",
                label="Smoking Status"
            )

            region = gr.Dropdown(
                [
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest"
                ],
                value="northeast",
                label="Region"
            )

            predict_btn = gr.Button(
                "🔮 Predict Insurance Cost",
                variant="primary",
                size="lg"
            )

        with gr.Column(elem_classes="card"):

            gr.Markdown("### 💰 Prediction")

            result = gr.Textbox(
                label="Estimated Insurance Cost",
                placeholder="Prediction will appear here...",
                elem_classes="result",
                interactive=False
            )

            gr.Markdown("""
### 📊 Model Performance

**Polynomial Regression — Degree 2**

- R² Score: **0.8825**
- RMSE: **$4,646.06**
- MSE: **21,585,843.72**
- MAE: **$2,867.32**

### 🤖 Features

- Age
- BMI
- Children
- Gender
- Smoking status
- Region
            """)

    predict_btn.click(
        predict_insurance,
        [age, bmi, children, gender, smoker, region],
        result
    )

    gr.Markdown(
        """
        <center>
        Medical Insurance Cost Prediction • Machine Learning Project
        </center>
        """
    )


app.launch()
