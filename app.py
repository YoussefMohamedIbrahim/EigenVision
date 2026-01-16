import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage

# Bindings import
import eigenvision

class EigenVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EigenVision Classifier (Debug Mode)")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        # State variables
        self.pca = None
        self.knn = None
        self.current_image_path = None
        self.processed_matrix = None
        
        self.create_widgets()
        self.load_models()

    def create_widgets(self):
        # Header
        header = tk.Label(self.root, text="Digit Recognizer", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=10)

        tk.Label(left_frame, text="Your Input Image", font=("Arial", 10, "bold")).pack()
        self.canvas = tk.Canvas(left_frame, width=200, height=200, bg="#e0e0e0", relief="sunken", bd=2)
        self.canvas.pack(pady=5)
        self.image_container = None 

        self.btn_load = tk.Button(left_frame, text="Select Image...", command=self.select_file, width=15)
        self.btn_load.pack(pady=10)

        self.btn_predict = tk.Button(left_frame, text="Predict", command=self.run_prediction, width=15, state="disabled")
        self.btn_predict.pack(pady=5)

        self.lbl_result = tk.Label(left_frame, text="Result: --", font=("Arial", 20, "bold"), fg="#333")
        self.lbl_result.pack(pady=20)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        tk.Label(right_frame, text="What the Model Sees (28x28)", font=("Arial", 10, "bold")).pack()
        
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.ax.axis('off')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack()

        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def load_models(self):
        model_dir = "models"
        pca_path = os.path.join(model_dir, "pca_model.bin")
        knn_path = os.path.join(model_dir, "knn_model.bin")

        try:
            self.pca = eigenvision.PCA()
            self.knn = eigenvision.KNN()
            if os.path.exists(pca_path) and os.path.exists(knn_path):
                self.pca.load(pca_path)
                self.knn.load(knn_path)
                self.status_var.set("Models loaded.")
            else:
                self.status_var.set("Models missing!")
                messagebox.showerror("Error", "Models not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_image(self, path):
        img = Image.open(path).convert('L') 
        
        if np.mean(img) > 127:
            img = ImageOps.invert(img)
            
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)
        
        temp_arr = np.array(img)
        
        if np.sum(temp_arr) == 0:
            cy, cx = 10, 10
        else:
            cy, cx = ndimage.center_of_mass(temp_arr)

        shift_y = 14 - cy
        shift_x = 14 - cx
        
        new_img = Image.new('L', (28, 28), 0)
        
        paste_x = int(shift_x)
        paste_y = int(shift_y)
        
        paste_x = max(0, min(paste_x, 28 - img.width))
        paste_y = max(0, min(paste_y, 28 - img.height))
        
        new_img.paste(img, (paste_x, paste_y))
        
        arr = np.array(new_img)
        
        arr[arr < 50] = 0 
        
        norm_arr = arr.astype(np.float32) / 255.0
        
        mat = eigenvision.Matrix(1, 784)
        flat = norm_arr.flatten()
        for i in range(784):
            mat[0, i] = flat[i]
            
        return norm_arr, mat

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.current_image_path = file_path
            
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.Resampling.NEAREST)
            self.image_container = ImageTk.PhotoImage(img)
            self.canvas.create_image(100, 100, image=self.image_container)
            
            debug_arr, _ = self.process_image(file_path)
            self.update_debug_view(debug_arr)

            self.btn_predict.config(state="normal")
            self.lbl_result.config(text="Result: --")
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")

    def update_debug_view(self, arr):
        self.ax.clear()
        self.ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        self.ax.axis('off')
        self.ax.set_title("Preprocessed Input")
        self.canvas_plot.draw()

    def run_prediction(self):
        if not self.current_image_path: return
        try:
            _, img_mat = self.process_image(self.current_image_path)
            reduced_mat = self.pca.transform(img_mat)
            prediction = self.knn.predict(reduced_mat, k=5)
            self.lbl_result.config(text=f"Result: {int(prediction)}")
            self.status_var.set("Done.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = EigenVisionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()