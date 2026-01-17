import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import eigenvision

pca = eigenvision.PCA()
knn = eigenvision.KNN()

print("Loading models...")
pca.load("models/pca_model.bin")
knn.load("models/knn_model.bin")
print("Models loaded.")

def predict_digit(image):
    if image is None:
        return "Draw something first!"
    

    img = Image.fromarray(image.astype('uint8')).convert('L')

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img = ImageOps.invert(img)

    arr = np.array(img)
    norm_arr = arr.astype(np.float32) / 255.0

    mat = eigenvision.Matrix(1, 784)
    flat = norm_arr.flatten()
    for i in range(784):
        mat[0, i] = flat[i]

    reduced_mat = pca.transform(mat)
    prediction = knn.predict(reduced_mat, k=5)

    return int(prediction)

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a Digit", type="numpy", crop_size=(200, 200)),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="EigenVision (C++ Backend)",
    description="Draw a digit (0-9). The C++ PCA/KNN backend will recognize it."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)