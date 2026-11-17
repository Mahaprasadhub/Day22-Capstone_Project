# Energy-Demand Forecasting with LSTM + Attention

## Project Overview

This capstone project builds an end-to-end multi-horizon energy-demand forecasting system using TensorFlow/Keras. The dataset is generated entirely in Python, so no external CSV file or URL is required.

The objective is to use the previous **168 hours (1 week)** of energy-demand observations to predict the next **24 hours (1 day)**.

Three forecasting approaches are compared:

1. Seasonal-naive baseline
2. LSTM forecaster
3. LSTM with attention

The models are evaluated using **MAE, RMSE, and MAPE**.

## Dataset

The project generates approximately two years of hourly energy-demand data with:

- Rising long-term trend
- Daily 24-hour seasonality
- Weekly 168-hour seasonality
- Lower weekend demand
- Random noise

The generated series contains **17,472 hourly observations**.

## Sliding Window

The time series is converted into a supervised-learning problem using sliding windows.

- `LOOKBACK = 168`: previous 168 hours are used as input.
- `HORIZON = 24`: next 24 hours are predicted.
- Stride = 1 hour.

This produces input windows with 168 values and target windows with 24 values.

## Data Preparation

The data is kept in chronological order and divided into:

- 70% training
- 15% validation
- 15% testing

The data is not shuffled because future observations must not be used to predict the past.

Normalization uses the mean and standard deviation calculated from the training inputs only, helping prevent future information from leaking into model training.

## Seasonal-Naive Baseline

The baseline assumes that the next 24 hours will resemble the last 24 hours in the input window.

Because the synthetic dataset contains strong daily seasonality, this provides a meaningful benchmark that the neural-network models should outperform.

## LSTM Model

The plain LSTM model contains:

- Input shape: `(168, 1)`
- LSTM layer with 48 units
- Dropout of 0.1
- Dense output layer with 24 units

The model predicts all 24 future values simultaneously, making it a direct multi-output forecasting model.

Training uses:

- Adam optimizer with learning rate `0.001`
- Mean Squared Error loss
- MAE metric
- Maximum 15 epochs
- Batch size 128
- Early stopping with patience 3

## LSTM with Attention

The attention model uses an LSTM with `return_sequences=True`, which keeps the hidden state for every input timestep.

The architecture includes:

- LSTM with 48 units
- Self-attention over encoder states
- Residual connection
- Layer normalization
- Global average pooling
- Dropout of 0.1
- Dense output layer with 24 units

Attention allows the network to use information from different parts of the 168-hour history instead of depending only on the final LSTM state.

## Evaluation Metrics

The three approaches are compared using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted demand.

### RMSE

Root Mean Squared Error gives larger errors more weight than MAE.

### MAPE

Mean Absolute Percentage Error expresses the average prediction error as a percentage of actual demand.

For all three metrics, **lower values indicate better forecasting performance**.

## Teacher Forcing vs Autoregressive Inference

The models in this project predict all 24 future values at once using a Dense output layer. Therefore, they do not use teacher forcing or autoregressive inference.

In an autoregressive seq2seq model, the previous true value may be supplied during training through teacher forcing, while the model must use its own previous prediction during inference. This train/inference mismatch can cause exposure bias and error accumulation.

## Project Structure

```text
.
├── day22_capstone.ipynb
└── README.md
```

## Requirements

Recommended environment:

- Python 3.x
- JupyterLab or Jupyter Notebook
- NumPy
- Matplotlib
- TensorFlow / Keras

Example installation:

```bash
pip install numpy matplotlib tensorflow jupyterlab
```

## How to Run

1. Open JupyterLab or Jupyter Notebook.
2. Open `day22_capstone.ipynb`.
3. Select a Python kernel containing TensorFlow.
4. Run the notebook cells from top to bottom.
5. Wait for both neural networks to finish training.
6. Compare the baseline, LSTM, and LSTM+attention metrics.
7. Inspect the 24-hour prediction plot.
8. Replace the bracketed placeholders in the final Analysis section with the actual MAE, RMSE, MAPE, and model-comparison values from your run.

To start JupyterLab from a terminal:

```bash
jupyter lab
```

## Expected Output

The notebook produces:

- Generated energy-demand series statistics
- Sliding-window dimensions
- Training, validation, and test dimensions
- Seasonal-naive baseline metrics
- LSTM training results
- LSTM+attention training results
- Comparison of MAE, RMSE, and MAPE
- Best model according to test MAE
- 24-hour forecast plot
- Metric differences against the baseline
- Final written analysis

## Key Learning Outcomes

This project demonstrates how to:

- Convert a time series into supervised samples with sliding windows.
- Preserve temporal order during train/validation/test splitting.
- Normalize data using training statistics only.
- Establish a seasonal-naive forecasting baseline.
- Build an LSTM time-series forecasting model.
- Add attention over LSTM encoder states.
- Perform multi-horizon forecasting.
- Evaluate forecasts using MAE, RMSE, and MAPE.
- Understand direct, teacher-forced, and autoregressive forecasting approaches.
- Compare model complexity against measurable test-set improvement.

## Conclusion

The project provides a complete sequence-model forecasting workflow: data generation, sliding-window preparation, chronological splitting, normalization, baseline comparison, LSTM modeling, attention, and final evaluation. The important result is not simply whether attention is more sophisticated, but whether its additional complexity produces a measurable improvement over both the plain LSTM and the strong seasonal-naive baseline.
