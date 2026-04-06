# MNIST Digit Classification with Saliency Maps 👁️🧠

This repository contains a technology-driven image classification application built with **PyTorch**. The project implements a custom Convolutional Neural Network (CNN) from scratch to accurately recognize and classify handwritten digits from the MNIST dataset. 

As part of the focus on Model Explainability in AI, gradient-based **Saliency Maps** are integrated to visualize exactly which pixels the neural network focuses on to make its predictions.

## 🚀 Key Features

* **Custom CNN Architecture:** A lightweight yet highly effective Convolutional Neural Network built utilizing `torch.nn`, featuring convolutional layers, max pooling, and fully connected layers.
* **High Performance:** The model achieved an outstanding **99.00% classification accuracy** on the 10,000 unseen test images.
* **AI Explainability:** Instead of treating the model as a "black box," the project uses Saliency Maps to provide a heatmap of pixel importance. This proves that the network actually learns the shapes of the digits rather than memorizing random noise.

## 🛠️ Technology Stack

* **Language:** Python 3
* **Deep Learning Framework:** PyTorch & Torchvision
* **Data Visualization:** Matplotlib, NumPy
* **Dataset:** MNIST (Handwritten Digits)

## 📊 Visual Results

During the testing phase, the application selects a test image, makes a prediction, and generates a three-part plot:
1. **Original Image:** The raw input fed to the model.
2. **Saliency Heatmap:** A visual representation of the gradients, showing the areas of highest influence on the prediction.
3. **Overlay:** The heatmap superimposed on the original image for clear interpretability.


## 💻 How to Run Locally

Follow these steps to clone the repository, install dependencies, and run the model on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/Emirrbozkurt/MNIST-CNN-Saliency-Map.git](https://github.com/Emirrbozkurt/MNIST-CNN-Saliency-Map.git)
cd MNIST-CNN-Saliency-Map
2. Install required packages:

Bash
pip install torch torchvision matplotlib numpy
3. Run the main application:

Bash
python main.py
The script will automatically download the MNIST dataset (if not already present), train the CNN for 5 epochs, evaluate the accuracy, and display the Saliency Map visualization.

Author: Emir Bozkurt

Institution: Istinye University

Context: Developed as a final project for the Introduction to Large Language Models (LLMs) course.


