# PyTorch FashionMNIST Computer Vision

This project explores computer vision using **PyTorch** on the **FashionMNIST** dataset.  
It demonstrates how to build, train, and evaluate multiple neural network models, starting from a simple baseline and progressing to a Convolutional Neural Network (CNN).

---

## 📌 Dataset
**FashionMNIST**
- 70,000 grayscale images (28×28)
- 10 fashion categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Loaded using `torchvision.datasets.FashionMNIST`

---

## 🧠 Models Implemented

### Model 0 – Baseline (Linear Model)
- Flatten layer
- Two fully connected (`Linear`) layers
- Used as a performance baseline

### Model 1 – Non-Linear Fully Connected Network
- Flatten layer
- Linear → ReLU → Linear → ReLU
- Improves learning using non-linearity

### Model 2 – Convolutional Neural Network (CNN)
- Two convolutional blocks with ReLU and MaxPooling
- Fully connected classifier
- Best performance on FashionMNIST

---

## ⚙️ Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib
- TQDM
- Google Colab (for training)

---

## 📊 Evaluation
- Loss Function: `CrossEntropyLoss`
- Optimizer: Stochastic Gradient Descent (SGD)
- Metric: Accuracy
- Training time tracked for each model
- Results compared using a Pandas DataFrame

---

## 📈 Results
Model performance and training time are compared across:
- Model accuracy
- Training duration
- Visualized using bar charts

---

## 🔮 Predictions
- Random test samples are passed to the best model
- Predictions are visualized alongside ground-truth labels
- Includes probability outputs and class predictions

---

## 🚀 How to Run
```bash
pip install torch torchvision matplotlib tqdm
