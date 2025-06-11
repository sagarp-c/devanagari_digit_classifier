# 🧐 Handwritten Devanagari Digit Classifier

A deep learning project that uses Convolutional Neural Networks (CNNs) to recognize handwritten Devanagari digits (०–९). This project was driven by my curiosity to explore advanced ML techniques and apply them to Indian scripts — an area that's rich with culture and complexity but often underserved.

---

## 🚀 Project Motivation

Inspired by the classic MNIST project, I wanted to go beyond Latin digits and explore Indian-language scripts — specifically Devanagari. I built this project to deepen my knowledge of CNNs, experiment with GPU training via Docker, and produce a practical classifier that can generalize well.

---

## 📁 Dataset

* A major part of the dataset was **hand-generated** by me using a custom-built digit drawing tool (`digit_drawer.html`).
* Additional reference: [Devanagari Handwritten Character Dataset (DHCD)](https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset)
* Final dataset size: **33,120 PNG files** across **10 digit classes (०–९)**, each with over 3,000 samples.
* Images were resized to either `64x64` or `28x28` depending on the model, and augmented using:

  * Rotation
  * Shear
  * Brightness/contrast variations
  * Noise injection

---

## 🏐 Model Architecture

Implemented using `TensorFlow + Keras`:

* 2× Conv2D layers with ReLU and L2 regularization
* MaxPooling + Dropout
* Fully connected Dense layer
* Final output layer with 10 logits

Compiled with:

```python
optimizer='adam',
loss=SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
```

---

## ⚙️ Training & Evaluation

Two separate models were trained:

* `final_model.keras`: trained on `28x28` grayscale input
* `prefinal_model.keras`: trained on `64x64` color input

Training was done for 20 epochs on RTX 3050 GPU using Docker.

From the **best\_model.keras** evaluation:

* **Validation Accuracy**: **97.55%**
* **Validation Loss**: **0.2109**

### 📊 Training Graphs (Matplotlib generated)

![Accuracy & Loss](training_performance.png)
![Model Performance](best_model_performance.png)
![Confusion Matrix](best_model_confusion_matrix.png)
![Detailed Metrics](best_model_detailed_metrics.png)

---

## 🔍 Test Script

Use `final_test.py` to test the model on any image:

```bash
python final_test.py
```

### 🔬 How to Test the Model

1. Place your test image inside the project directory.
2. Ensure the image is preprocessed to match the input size of the model (either 28×28 or 64×64 depending on which model you're testing).
3. Run the script:

   ```bash
   python final_test.py --img path_to_your_image.png --model best_model.keras
   ```
4. The script will print the predicted digit along with model confidence.

You can also use images generated from the `digit_drawer.html` tool for testing.

---

## 🖌️ Interactive Input Tool

Custom-built HTML-based digit drawing tool `digit_drawer.html`:

* Save variations like noise, tilt, shape-based strokes
* Automatically formats to 28x28 and saves to local disk

---

## 🐳 Run with Docker

You can run this project in a clean GPU container:

```bash
docker run --gpus all -it -v ${PWD}:/app -w /app tensorflow/tensorflow:2.15.0-gpu bash
python final_train.py
```

---

## 📚 Skills Demonstrated

* Deep Learning (CNNs)
* Dataset preprocessing & augmentation
* TensorFlow/Keras training pipelines
* Docker-based GPU acceleration
* Matplotlib visualizations
* Image classification metrics & evaluation
* Custom data generation via browser app

---

## 🔧 Areas for Improvement

* Improve generalization by expanding real-world handwritten data (especially from different age groups and writing styles).
* Integrate batch normalization to stabilize deeper models.
* Experiment with modern architectures like MobileNet or EfficientNet.
* Extend the UI to allow real-time prediction directly in-browser using TensorFlow\.js.
* Train a unified model that supports both `28x28` and `64x64` input formats.
* Build a REST API or Gradio demo to make the model more accessible.

---

## 👻 Resume Highlights

* ✅ Developed and trained CNN-based classifiers (28x28 and 64x64) for Devanagari digits using TensorFlow/Keras, achieving **up to 99.5% accuracy**.
* ✅ Built a complete GPU-accelerated training pipeline using Docker and visualized performance using Matplotlib, confusion matrices, and classification metrics.
* ✅ Designed an interactive HTML5 drawing tool to generate synthetic handwritten data with variations like brush shape, noise, and tilt.

---

## 📓 License

This project is released under the [MIT License](LICENSE).

---

## 🙆‍♂️ Author

**Sagar**
B.Tech CSE @ GEC Thrissur
Curious about ML, GPU computing, and building useful AI tools.

[LinkedIn](#) | [GitHub](#)
