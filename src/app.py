import numpy as np
import gradio as gr
import joblib
import os

# Load pre-trained models using joblib
knn_model = joblib.load('/workspaces/Iris-flower-classification/models/iris_knn_model.pkl')
svm_model = joblib.load('/workspaces/Iris-flower-classification/models/iris_svm_model.pkl')
logreg_model = joblib.load('/workspaces/Iris-flower-classification/models/iris_logreg_model.pkl')

# Dictionary mapping model names to models
model_dict = {
    "Logistic Regression": logreg_model,
    "K-Nearest Neighbors (KNN)": knn_model,
    "Support Vector Machine (SVM)": svm_model
    
}

# Mapping of prediction values to iris species and the corresponding image file path.
iris_classes = {
    0: ("Setosa", "/workspaces/Iris-flower-classification/img/flower_img/Iris_setosa.jpg"),
    1: ("Versicolor", "/workspaces/Iris-flower-classification/img/flower_img/Iris_versicolor.jpg"),
    2: ("Virginica", "/workspaces/Iris-flower-classification/img/flower_img/Iris_virginica.jpg")
}

def predict_iris(sepal_length, sepal_width, petal_length, petal_width, selected_model):
    print("Input values:", sepal_length, sepal_width, petal_length, petal_width, selected_model)
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    print("Features array:", features)
    
    model = model_dict[selected_model]
    prediction = model.predict(features)[0]
    print("Raw prediction from model:", prediction)
    
    # Unpack species info (ignore extra items if any)
    species_info = iris_classes[prediction]
    species_name, image_path, *_ = species_info
    print("Mapped species info:", species_name, image_path)
    
    # Verify that the image exists
    if not os.path.exists(image_path):
        print("Image file not found at:", image_path)
        image_path = None
    else:
        print("Image found:", image_path)
        
    prediction_text = f"Predicted Iris Species: {species_name}"
    print("Returning output:", prediction_text, image_path)
    return prediction_text, image_path

# Define the Gradio input components
inputs = [
    gr.Number(label="Sepal Length (cm)", value=5.4, precision=2),
    gr.Number(label="Sepal Width (cm)", value=3.4, precision=2),
    gr.Number(label="Petal Length (cm)", value=4.5, precision=2),
    gr.Number(label="Petal Width (cm)", value=1.3, precision=2),
    gr.Dropdown(choices=list(model_dict.keys()), label="Select Model", value="K-Nearest Neighbors (KNN)")
]

# Define the Gradio output components: one for text, one for the image.
outputs = [
    gr.Textbox(label="Prediction Output"),
    gr.Image(label="Flower Image", type="filepath")
]

# Footer HTML for LinkedIn and GitHub profiles
footer_html = """
<footer style="text-align: center; margin-top: 20px; font-family: Arial, sans-serif;">
  <p>Developed ❤️ with Gradio by DINESH S.</p>
  <div style="display: inline-flex; align-items: center; justify-content: center; gap: 10px; margin-top: 10px;">
    <h3>Connect with me:</h3>
    <a href="https://www.linkedin.com/in/dinesh-x/" target="_blank">
      <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" style="width:32px;">
    </a>
    <br>
    <a href="https://github.com/itzdineshx/Iris-flower-classification" target="_blank">
      <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width:32px;">
    </a>
  </div>
</footer>
"""

interface = gr.Interface(
    fn=predict_iris,
    inputs=inputs,
    outputs=outputs,
    title="Iris Flower Species Predictor🌸",
    description=(
        "Provide the measurements of the iris flower and select a model to predict its species. "
        "The app will display the predicted species along with a representative image."
    ),
    article=(
        "Adjust the inputs to see the predicted iris species and a related image."
        + footer_html
    )
)

# Launch the Gradio app and add the allowed_paths parameter for the image directory.
interface.launch(share=True, allowed_paths=["/workspaces/Iris-flower-classification/img/flower_img/"])
