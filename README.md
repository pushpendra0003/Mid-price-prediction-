# Mid-Price Prediction Using Deep Learning on Limit Order Book Data

## Project Motivation

Modern financial markets generate vast amounts of high-frequency limit order book (LOB) data. Predicting short-term price movements (mid-price) from this data is a crucial challenge for algorithmic trading and market making. This project demonstrates a complete pipeline: from raw LOB data processing, advanced feature engineering, target construction for forecasting, to deep learning modeling and trading strategy evaluation.

## Workflow Summary

1. **Data Conversion:** 
   - Raw `.h5` files (high-frequency order book snapshots) are converted to `.csv` for analysis.
   - See `src/data_transform.py`.

2. **Feature Engineering:**  
   - Extracts high-value features from LOB, such as price/volume imbalances, spreads, VWAP, and temporal features.
   - Converts time from milliseconds to standard datetime format.
   - See `src/feature_engineering.py`.

3. **Update Type Encoding:**  
   - Encodes the type of order book update (order, cancel, trade, etc.) as one-hot features for ML compatibility.
   - See `src/update_type_encoded.py`.

4. **Forecast Horizon & Target Construction:**  
   - Adds future mid-price columns for multiple forecast horizons (30s, 1min, ... 1hr) and lagged features for time-series modeling.
   - See `src/forecast_horizons.py`.

5. **Deep Learning Model:**  
   - Implements a sophisticated multi-input neural network.  
   - Uses LSTM, Conv1D, attention, and feature-wise processing for robust prediction and confidence estimation.
   - Handles class imbalance, advanced callbacks, and mixed precision for efficient GPU utilization.
   - See `src/Final_classifier.py`.

6. **Trading Simulation:**  
   - Realistic backtesting using model predictions and confidence scores.
   - Advanced position sizing, stop-loss, and performance metrics (Sharpe, Sortino, Calmar, drawdown, win rate).
   - Plots and saves detailed results for interview review.
   - See `src/trading_simulator.py`.

## Key Features & Innovations

- **Modular, error-resistant scripts:** Each stage is isolated for clarity and reproducibility.
- **Advanced Feature Engineering:** Volume/price imbalances, spread, VWAP, lagged/future targets.
- **Multi-Horizon Forecasting:** Targets for several future intervals allow flexible training and evaluation.
- **State-of-the-art Model:** Multi-input design with LSTM, Conv1D, multi-head attention, and advanced confidence scoring.
- **Robust Trading Simulation:** Implements real-world trading constraints and risk management, not just naive backtesting.
- **Clear Logging & Visualization:** All important metrics, confusion matrices, ROC/PR curves, confidence analysis, and performance tables are plotted and saved.

## Repository Structure

```
Mid-price-prediction-/
  ├── src/
  │     ├── data_transform.py
  │     ├── feature_engineering.py
  │     ├── update_type_encoded.py
  │     ├── forecast_horizons.py
  │     ├── Final_classifier.py
  │     └── trading_simulator.py
  ├── data/          # Add raw .h5 files here (not included)
  ├── results/       # Model outputs, logs, plots
  ├── docs/          # Documentation, PPT, project overview
  ├── requirements.txt
  ├── .gitignore
  └── README.md
```

## How to Use

1. **Prepare Data:**  
   Place your raw `.h5` stock order book files in the `data/` folder.

2. **Run Pipeline:**  
   Execute scripts in order in the `src/` directory:
   - `data_transform.py` → `feature_engineering.py` → `update_type_encoded.py` → `forecast_horizons.py` → `Final_classifier.py`

3. **Evaluate Model:**  
   - After training, run `trading_simulator.py` to simulate trading and generate results in the `results/` folder.

4. **Review Results:**  
   - Check `results/` for metrics, visuals, and performance tables.
   - For a project overview, see the PPT in `docs/` (add your file as needed).

## File Descriptions

- `src/data_transform.py`: Inspects and converts `.h5` files into `.csv`, preserving all order book details.
- `src/feature_engineering.py`: Adds essential trading features (imbalance, spread, VWAP, EMA, time features) and ensures chronological order.
- `src/update_type_encoded.py`: One-hot encodes order update types; saves encoded data for modeling.
- `src/forecast_horizons.py`: Constructs future target columns for forecasting (multi-horizon), adds lags, and ensures time alignment.
- `src/Final_classifier.py`: Defines, trains, and evaluates a deep learning classifier with multi-input architecture, confidence scoring, and robust cross-validation.
- `src/trading_simulator.py`: Simulates trading using predicted signals and confidence, applies risk management, and visualizes performance.

## For Interviewers

- All scripts are modular, well-commented, and error-handled.
- Project demonstrates advanced ML, feature engineering, and practical trading simulation.
- Reproducible, with all outputs saved for review.
- Please see the `docs/` folder for my BTP presentation and further project background.

---

**Author:** Pushpendra Jain  
**Project Type:** BTP (Bachelor Thesis Project)  
**Contact:** [GitHub](https://github.com/pushpendra0003)
