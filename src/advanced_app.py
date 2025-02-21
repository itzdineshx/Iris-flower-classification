import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

# Function to safely load models
def load_model(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        print(f"⚠️ Warning: {filename} not found.")
        return None

# Load all trained models and utilities
knn = load_model('knn_model.joblib')
svm = load_model('svm_model.joblib')
logreg = load_model('logreg_model.joblib')
dtc = load_model('dtc_model.joblib')
rfc = load_model('rfc_model.joblib')
scaler = load_model('scaler.joblib')
le = load_model('label_encoder.joblib')

# Ensure all models are loaded before proceeding
if None in [knn, svm, logreg, dtc, rfc, scaler, le]:
    raise RuntimeError("🚨 Error: One or more model files are missing. Please check and reload.")

# Dictionary to map species names to image file paths
iris_images = {
    "Iris-setosa": "/content/Iris_setosa.jpg",
    "Iris-versicolor": "/content/Iris_versicolor.jpg",
    "Iris-virginica": "/content/Iris_virginica.jpg"
}

# Function to plot the flower image
def plot_flower(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    else:
        # Return a placeholder or blank image if not found
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.text(0.5, 0.5, "🌸 No Image Available", fontsize=12, ha="center")
        ax.axis("off")
        plt.show()
        return None

# Prediction function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width, model_choice):
    # Validate input values (no negative numbers)
    if any(v <= 0 for v in [sepal_length, sepal_width, petal_length, petal_width]):
        return "🚨 Error: All values must be positive numbers.", None
    
    # Convert input to NumPy array
    custom_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Convert to DataFrame with correct feature names
    custom_input_df = pd.DataFrame(custom_input, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    # Scale the input using StandardScaler
    custom_input_scaled = scaler.transform(custom_input_df)

    # Select model and make prediction
    if model_choice == "KNN":
        prediction = knn.predict(custom_input_scaled)
    elif model_choice == "SVM":
        prediction = svm.predict(custom_input_scaled)
    elif model_choice == "Logistic Regression":
        prediction = logreg.predict(custom_input_scaled)
    elif model_choice == "Decision Tree":
        prediction = dtc.predict(custom_input_scaled)
    elif model_choice == "Random Forest":
        prediction = rfc.predict(custom_input_scaled)
    else:
        return "🚨 Invalid model choice!", None
    
    # Get predicted species name
    predicted_species = le.inverse_transform(prediction)[0]

    # Load and return the related image
    image_path = iris_images.get(predicted_species, None)
    
    return predicted_species, plot_flower(image_path)


# Footer HTML for LinkedIn and GitHub profiles
footer_html = """
<footer style="text-align: center; margin-top: 20px; font-family: Arial, sans-serif;">
  <p>Developed ❤️ with Gradio by DINESH S.</p>
  <div style="display: inline-flex; align-items: center; justify-content: center; gap: 10px; margin-top: 10px;">
    <h3>Connect with me:</h3>
    <a href="https://www.linkedin.com/in/dinesh-x/" target="_blank">
      <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" style="width:32px;">
    </a>
    <a href="https://github.com/itzdineshx/Iris-flower-classification" target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width:32px;">
    </a>
    <a href="mailto:personalaccdinesh@gmail.com" target="_blank">
      <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" alt="Gmail" style="width:32px;">
    </a>

  </div>
</footer>
""" 

# Gradio Interface
iface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal Length (cm)", value=5.1),
        gr.Number(label="Sepal Width (cm)", value=3.5),
        gr.Number(label="Petal Length (cm)", value=1.4),
        gr.Number(label="Petal Width (cm)", value=0.2),
        gr.Dropdown(["KNN", "SVM", "Logistic Regression", "Decision Tree", "Random Forest"], 
                    label="Choose a Model", value="KNN")
    ],
    outputs=[
        gr.Textbox(label="Predicted Iris Species"),
        gr.Image(label="Flower Image")
    ],
    title="🌸 Iris Flower Species Prediction",
    description='<div style="text-align: center;">Enter the sepal and petal measurements, choose a model, and get the predicted species along with an image!</div>',

    examples=[
    [5.1, 3.5, 1.4, 0.2, "Random Forest"],# Typical Iris-setosa
    [6.0, 2.9, 4.5, 1.5, "SVM"],  # Typical Iris-versicolor
    [6.7, 3.1, 5.6, 2.4, "KNN"], # Typical Iris-virginica
    ],

    article=footer_html,
    css=".gr-description { text-align: center; }"

)

iface.launch()
