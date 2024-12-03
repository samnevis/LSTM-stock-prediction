# Stock Price Prediction Using LSTM

This project demonstrates the use of Long Short-Term Memory (LSTM) networks for predicting Google stock prices based on historical data. The model is implemented using TensorFlow and evaluates the predictions using multiple metrics such as Mean Absolute Percentage Error (MAPE), Directional Accuracy (DA), and Simulated Trading Profits.

---



## Overview

The project uses an LSTM model to predict Google stock prices based on their closing values over a given period. It explores data preprocessing, model training, and evaluation on distinct datasets (training, validation, and testing).

Key objectives:
- Build a robust time-series model using LSTM.
- Evaluate the model's predictive capabilities using MAPE and DA.
- Simulate trading strategies based on model predictions to evaluate financial implications.

---

## Dataset

The dataset, **Google Stock Price (2010-2023)**, is sourced from Kaggle and includes daily stock prices with fields like `Date`, `Open`, `High`, `Low`, and `Close`. This project uses the `Close` prices for modeling.

- Link to dataset: [Google Stock Price Dataset](https://www.kaggle.com/datasets/alirezajavid1999/google-stock-2010-2023)

---

## Dependencies

To run this project, ensure you have the following Python libraries installed:

- TensorFlow
- NumPy
- Pandas
- Matplotlib

Install all dependencies with:
```bash
pip install tensorflow numpy pandas matplotlib
```

---

## Preprocessing

The data is preprocessed as follows:
1. **Data Segmentation**: The dataset is split into training (80%), validation (10%), and testing (10%) sets.
2. **Sliding Window**: A sliding window approach is used to create input-output pairs for the model. For a window size of 5, the past 5 days' prices predict the next day's price.
3. **Normalization**: Data normalization ensures consistent scaling, improving LSTM performance.

---

## Model Architecture

The LSTM model is built using TensorFlow's Keras API with the following layers:
- **Input Layer**: Accepts sequences of shape `(5, 1)` (5 days of stock prices).
- **LSTM Layer**: Contains 64 units to capture temporal dependencies.
- **Dense Layers**: Two fully connected layers with 32 units each and ReLU activation.
- **Output Layer**: A single neuron for predicting the stock price.

Optimizer: `Adam` with a learning rate of 0.001.

Loss Function: Mean Squared Error (`mse`).

---

## Training and Evaluation

The model is trained over 20 epochs with a batch size of 32. The loss and validation error are monitored to ensure model convergence.

**Datasets**:
- **Training Set**: Used to train the model.
- **Validation Set**: Used to tune hyperparameters and prevent overfitting.
- **Testing Set**: Used to evaluate the model's performance on unseen data.

---

## Metrics

### Mean Absolute Percentage Error (MAPE)

MAPE measures the model's prediction accuracy:
\[
\text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{\text{Actual}_i - \text{Predicted}_i}{\text{Actual}_i} \right| \times 100
\]

### Directional Accuracy (DA)

DA evaluates how well the model predicts the direction of price movement:
\[
\text{DA} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100
\]

### Simulated Trading Profits

This metric simulates a simple trading strategy:
- **Buy** if the predicted price increases.
- **Sell** if the predicted price decreases.

Cumulative profits are calculated based on the predicted direction's accuracy relative to actual price changes.

---

## Results

- **MAPE**: Measures the error in predictions as a percentage of actual values.
- **Directional Accuracy**: Percentage of times the predicted price direction matches the actual direction.
- **Simulated Profit**: Total profit generated using the model's predictions in a simulated trading strategy.

```
MAPE: 2.6 (97.4% accurate)
Directional Accuracy: 55.0%
Cumulative Profit: -$6.1
```

---
