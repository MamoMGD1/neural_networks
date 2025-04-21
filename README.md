# 🏠 House Price Predictor – From Scratch with Neural Networks

Welcome to my machine learning project where I built a **neural network from scratch** to predict house prices based on real-world housing data.  
This repository contains everything from **data preprocessing**, **visualization**, **model training**, to **evaluation**, without relying on high-level machine learning libraries like TensorFlow or PyTorch.

## 📂 Contents

- `data.csv` – Real-world dataset of 20,000+ houses with features like area, number of rooms, house's age, etc.
- `main.ipynb` – Jupyter Notebook containing all code for data cleaning, visualizations, model implementation, training, and testing.

---

## 🤖 Why Neural Networks?

A **neural network** is a type of machine learning model inspired by the structure of the human brain. It processes input features through layers of interconnected "neurons" that apply weights and activation functions to capture complex relationships.

In this project, I used a neural network because:

- It can **model nonlinear relationships** between features and prices.
- It works well even when **correlations aren't obvious**.
- It's a powerful learning algorithm that can improve as more data is fed into it.

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   The raw housing dataset was messy. I wrote code to:
   - Handle missing values
   - Normalize numerical features
   - Prepare input features and target values for training

2. **Data Visualization**
   - A **correlation heatmap** was generated to better understand how each feature relates to the house price.
   - This gave me insight into which variables are most impactful.

3. **Neural Network Implementation (from scratch!)**
   - Fully implemented in NumPy (no high-level libraries used)
   - Single hidden-layer feedforward network
   - Supports any input size and scales dynamically based on dataset
   - Uses **ReLU activation**, **mean squared error (MSE)** loss, and **gradient descent**

4. **Model Training**
   - Trained the model on the cleaned dataset for 4200 epochs
   - Tracked training loss over time

5. **Evaluation & Results**
   - Achieved a final loss around **0.3**, which corresponds to **~70% accuracy**
   - Visualized model performance with a **bar plot of prediction errors**
   - The model successfully learned the difference between **cheap** and **expensive** houses based on their features

---

## 📉 Final Loss & Prediction Quality

Although a loss of 0.3 (MSE) doesn’t make this a production-grade model, it still demonstrates that a **simple neural network can learn useful patterns from real-world data**.

Predicted values are **reasonably close to actual prices**, and the network clearly distinguishes between homes with different characteristics.

---

## 💡 What I Learned

- How to clean and preprocess large, real-world datasets
- How to design and implement a neural network **from scratch**
- How to visualize and interpret correlations in data
- The importance of normalization in training stability
- That training accuracy isn't everything — **interpretability and generalization matter**

---

## 🧠 Future Plans

- Add support for multiple hidden layers
- Implement early stopping and batch training
- Compare performance with scikit-learn or TensorFlow versions
- Add a GUI or web dashboard for interactive price prediction
