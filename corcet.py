import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging
import seaborn as sns
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data (only 2021–2024)
try:
    ar_2021 = np.array(
        [21.79117167, 21.75852971, 22.39489592, 22.63133539, 22.65938101, 21.82767094, 20.66170384, 20.13220434,
         19.72630264, 18.91702654, 18.42084514, 18.21607675])
    ar_2022 = np.array(
        [16.42575, 16.67524, 17.3345, 17.74501, 17.50593, 17.4531, 16.60302, 16.63204, 16.77578, 16.83857, 16.2442,
         16.3977])
    ar_2023 = np.array(
        [15.64197, 16.34811, 17.14569, 17.48804, 17.97377, 17.67404, 16.24335, 16.20684, 16.18681, 16.34306, 15.73323,
         15.97414])
    ar_2024 = np.array(
        [14.34575, 14.8304, 15.40556, 15.9593, 16.18928, 15.54587, 13.87992, 13.86477, 13.88244, 13.76272, 13.31982,
         13.35429])

    actual = np.array([
        21.68, 21.68, 22.60, 22.71, 22.50, 20.75, 19.32, 18.18, 17.16, 16.75, 16.43, 16.34,  # 2021
        16.34, 16.34, 17.11, 17.36, 16.98, 16.60, 16.09, 16.34, 16.34, 16.21, 15.57, 15.43,  # 2022
        15.53, 16.10, 16.59, 16.97, 17.69, 17.43, 15.60, 15.43, 15.26, 15.29, 14.68647, 14.49,  # 2023
        14.34923, 14.7223, 15.17152, 15.78062, 15.94812, 14.91265, 12.99397, 12.48384, 12.41532, 12.43816, 12.2402, 12.26  # 2024
    ])
    actual_2021 = actual[:12]
    actual_2022 = actual[12:24]
    actual_2023 = actual[24:36]
    actual_2024 = actual[36:48]
    actual_train = np.concatenate([actual_2021, actual_2022, actual_2023])
    ar_train = np.concatenate([ar_2021, ar_2022, ar_2023])
    logging.info("Data loaded successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Calculate errors for each month (2021–2023)
errors = {}
try:
    for year, ar, act in zip([2021, 2022, 2023], [ar_2021, ar_2022, ar_2023], [actual_2021, actual_2022, actual_2023]):
        errors[year] = ar - act
    logging.info("Errors calculated successfully")
except Exception as e:
    logging.error(f"Error calculating errors: {e}")
    raise

# Prepare training data for each month (1–12)
years = np.array([2021, 2022, 2023]).reshape(-1, 1)
ar_values = np.array([ar_2021, ar_2022, ar_2023])  # Shape: (3, 12)
months = np.array([list(range(1, 13))] * 3).T  # Shape: (12, 3)
X_2024 = np.array([[2024, ar_2024[i], i + 1] for i in range(12)])  # Year, AR value, month index

# Initialize models
models = {
    'GaussianProcess': None,  # Use 2023 error
    'Mean Error': None,
    'Prophet': None,
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=20, random_state=42, min_samples_split=2, min_samples_leaf=2),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=20, random_state=42, max_depth=2, min_samples_split=2, min_samples_leaf=2),
    'XGBoost': XGBRegressor(n_estimators=20, random_state=42, max_depth=2, min_child_weight=2),
    'SVR': SVR(kernel='rbf', C=0.5, epsilon=0.05)
}

# Calculate NSE
def calculate_nse(actual, pred):
    mean_actual = np.mean(actual)
    numerator = np.sum((actual - pred) ** 2)
    denominator = np.sum((actual - mean_actual) ** 2)
    nse = 1 - numerator / denominator if denominator != 0 else 0
    logging.debug(f"NSE calculation: mean_actual={mean_actual:.4f}, numerator={numerator:.4f}, denominator={denominator:.4f}, NSE={nse:.4f}")
    return nse

# Store corrected predictions and metrics
corrected_predictions = {}
corrected_predictions_train = {}  # For 2021–2023
metrics_2024 = {}
metrics_train = {}
yearly_models = {}  # Initialize globally

# Train and predict for each model
try:
    for name, model in models.items():
        corrected = np.zeros(12)  # For 2024
        corrected_train = np.zeros(36)  # For 2021–2023 (3 years x 12 months)
        logging.info(f"Processing model: {name}")

        if name in ['Linear Regression', 'Gradient Boosting', 'XGBoost']:
            # Train one model per year for Linear Regression, Gradient Boosting, and XGBoost
            for year in [2021, 2022, 2023]:
                y_train = errors[year]  # 12 months of errors
                X_train = np.column_stack((np.full(12, year), ar_values[years.flatten().tolist().index(year), :], np.arange(1, 13)))  # Year, AR values, month indices
                current_model = models[name]  # Get the model instance
                current_model.fit(X_train, y_train)
                if year not in yearly_models:
                    yearly_models[year] = {}
                yearly_models[year][name] = current_model
                logging.debug(f"Trained {name} for year {year}")

            # Predict for 2024 using 2023 model
            for month in range(12):
                predicted_error_2024 = yearly_models[2023][name].predict(X_2024[month].reshape(1, -1))[0]
                corrected[month] = ar_2024[month] - predicted_error_2024
                logging.debug(f"Month {month + 1} (2024): predicted_error={predicted_error_2024:.4f}, corrected={corrected[month]:.4f}")

                # Predict for 2021–2023
                for i, year in enumerate([2021, 2022, 2023]):
                    idx = i * 12 + month
                    X_train_point = np.array([year, ar_values[i, month], month + 1]).reshape(1, -1)
                    predicted_error_train = yearly_models[year][name].predict(X_train_point)[0]
                    corrected_train[idx] = ar_values[i, month] - predicted_error_train
                    logging.debug(f"Year {year}, Month {month + 1}: predicted_error={predicted_error_train:.4f}, corrected={corrected_train[idx]:.4f}")

        else:
            # Original logic for other models
            for month in range(12):
                y_train = np.array([errors[year][month] for year in [2021, 2022, 2023]])
                X_train = np.column_stack((years, ar_values[:, month], months[month]))  # Year, AR value, month index
                logging.debug(f"Month {month + 1}: y_train={y_train}, X_train={X_train}")
                if name == 'GaussianProcess':
                    predicted_error_2024 = errors[2023][month]  # Use 2023 error for 2024
                    predicted_errors_train = [errors[2023][month] for year in [2021, 2022, 2023]]
                elif name == 'Mean Error':
                    predicted_error_2024 = np.mean(y_train)
                    predicted_errors_train = [np.mean(y_train) for year in [2021, 2022, 2023]]
                elif name == 'Prophet':
                    predicted_error_2024 = np.median(y_train)
                    predicted_errors_train = [np.median(y_train) for year in [2021, 2022, 2023]]
                else:
                    model.fit(X_train, y_train)
                    predicted_error_2024 = model.predict(X_2024[month].reshape(1, -1))[0]
                    predicted_errors_train = [model.predict(np.array([year, ar_values[years.flatten().tolist().index(year), month], month + 1]).reshape(1, -1))[0] for year in [2021, 2022, 2023]]

                # 2024 predictions
                corrected[month] = ar_2024[month] - predicted_error_2024
                logging.debug(f"Month {month + 1} (2024): predicted_error={predicted_error_2024:.4f}, corrected={corrected[month]:.4f}")

                # 2021–2023 predictions
                for i, year in enumerate([2021, 2022, 2023]):
                    idx = i * 12 + month
                    corrected_train[idx] = ar_values[i, month] - predicted_errors_train[i]
                    logging.debug(f"Year {year}, Month {month + 1}: predicted_error={predicted_errors_train[i]:.4f}, corrected={corrected_train[idx]:.4f}")

        corrected_predictions[name] = corrected
        corrected_predictions_train[name] = corrected_train
        metrics_2024[name] = {
            'MAE': mean_absolute_error(actual_2024, corrected),
            'RMSE': np.sqrt(mean_squared_error(actual_2024, corrected)),
            'NSE': calculate_nse(actual_2024, corrected)
        }
    logging.info("Models trained and predictions made successfully")
except Exception as e:
    logging.error(f"Error in model training or prediction: {e}")
    raise

# Debug yearly_models contents
logging.debug("Yearly models for 2023: %s", {name: type(model) for name, model in yearly_models.get(2023, {}).items()})

# Original metrics for 2024
metrics_2024['Original'] = {
    'MAE': mean_absolute_error(actual_2024, ar_2024),
    'RMSE': np.sqrt(mean_squared_error(actual_2024, ar_2024)),
    'NSE': calculate_nse(actual_2024, ar_2024)
}

# Compute metrics for 2021–2023
model_names = ['Original', 'GaussianProcess', 'Mean Error', 'Prophet',
               'Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'SVR']
metrics_train['Original'] = {
    'MAE': mean_absolute_error(actual_train, ar_train),
    'RMSE': np.sqrt(mean_squared_error(actual_train, ar_train)),
    'NSE': calculate_nse(actual_train, ar_train)
}
for name in model_names[1:]:
    corrected = corrected_predictions_train[name]
    metrics_train[name] = {
        'MAE': mean_absolute_error(actual_train, corrected),
        'RMSE': np.sqrt(mean_squared_error(actual_train, corrected)),
        'NSE': calculate_nse(actual_train, corrected)
    }

# Print results
print("Evaluation Metrics for 2024:")
for name in model_names:
    print(f"\n{name}:")
    print(f"MAE: {metrics_2024[name]['MAE']:.4f}")
    print(f"RMSE: {metrics_2024[name]['RMSE']:.4f}")
    print(f"NSE: {metrics_2024[name]['NSE']:.4f}")

print("\nEvaluation Metrics for 2021–2023:")
for name in model_names:
    print(f"\n{name}:")
    print(f"MAE: {metrics_train[name]['MAE']:.4f}")
    print(f"RMSE: {metrics_train[name]['RMSE']:.4f}")
    print(f"NSE: {metrics_train[name]['NSE']:.4f}")

# Plotting predictions (2024 only)
try:
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 13), actual_2024, label='Actual', color='blue', linewidth=2, marker='o')
    plt.plot(range(1, 13), ar_2024, label='Original Prediction', color='red', linestyle='--', marker='x')
    # Extend colors and styles to 8 elements to include all models
    colors = ['limegreen', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'gray', 'pink']
    styles = ['-', '-.', ':', '-', '--', '-.', ':', '--']
    for (name, corrected), color, style in zip(corrected_predictions.items(), colors, styles):
        plt.plot(range(1, 13), corrected, label=name, color=color, linestyle=style, marker='s')
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title('2024 Predictions: Actual vs Original vs Corrected (Month-Specific Correction)')
    plt.ylim(0, None)  # Set Y-axis to start at 0
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot.png')  # Save plot
    plt.show()
    logging.info("Prediction plot generated and saved as plot.png")
except Exception as e:
    logging.error(f"Error in plotting predictions: {e}")
    raise

# Plotting combined predictions (2021–2023 and 2024)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]}, sharey=False)

    # Extend colors and styles to 8 elements to include all models
    colors = ['limegreen', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'gray', 'pink']
    styles = ['-', '-.', ':', '-', '--', '-.', ':', '--']

    # Left subplot: 2021–2023
    train_years = np.arange(36)  # 36 months (Jan 2021–Dec 2023)
    ax1.plot(train_years, actual_train, label='Actual', color='blue', linewidth=2, marker='o')
    ax1.plot(train_years, ar_train, label='Original Prediction', color='red', linestyle='--', marker='x')
    for (name, corrected), color, style in zip(corrected_predictions_train.items(), colors, styles):
        ax1.plot(train_years, corrected, label=name, color=color, linestyle=style, marker='s')

    # Compute Y-axis limits for left subplot
    all_train_values = np.concatenate([actual_train, ar_train] + [corr for corr in corrected_predictions_train.values()])
    min_train = np.min(all_train_values)
    max_train = np.max(all_train_values)
    margin_train = (max_train - min_train) * 0.05  # 5% margin
    ax1.set_ylim(min_train - margin_train, max_train + margin_train)

    ax1.set_xlabel('Date (2021–2023)')
    ax1.set_ylabel('Value')
    ax1.set_title('2021–2023 Predictions: Actual vs Original vs Corrected')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(np.arange(0, 36, 12))
    ax1.set_xticklabels(['2021', '2022', '2023'])

    # Right subplot: 2024
    ax2.plot(range(1, 13), actual_2024, label='Actual', color='blue', linewidth=2, marker='o')
    ax2.plot(range(1, 13), ar_2024, label='Original Prediction', color='red', linestyle='--', marker='x')
    for (name, corrected), color, style in zip(corrected_predictions.items(), colors, styles):
        ax2.plot(range(1, 13), corrected, label=name, color=color, linestyle=style, marker='s')

    # Compute Y-axis limits for right subplot
    all_2024_values = np.concatenate([actual_2024, ar_2024] + [corr for corr in corrected_predictions.values()])
    min_2024 = np.min(all_2024_values)
    max_2024 = np.max(all_2024_values)
    margin_2024 = (max_2024 - min_2024) * 0.05  # 5% margin
    ax2.set_ylim(min_2024 - margin_2024, max_2024 + margin_2024)

    ax2.set_xlabel('Month (2024)')
    ax2.set_ylabel('Value')
    ax2.set_title('2024 Predictions: Actual vs Original vs Corrected')
    ax2.grid(True)
    ax2.legend()

    # Log Y-axis limits for verification
    logging.debug(f"Left subplot Y-axis limits: {ax1.get_ylim()}")
    logging.debug(f"Right subplot Y-axis limits: {ax2.get_ylim()}")

    plt.tight_layout()
    plt.savefig('combined_predictions.png')
    plt.show()
    logging.info("Combined predictions plot generated and saved as combined_predictions.png")
except Exception as e:
    logging.error(f"Error in plotting combined predictions: {e}")
    raise

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging
import seaborn as sns
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data (only 2021–2024)
try:
    ar_2021 = np.array(
        [21.79117167, 21.75852971, 22.39489592, 22.63133539, 22.65938101, 21.82767094, 20.66170384, 20.13220434,
         19.72630264, 18.91702654, 18.42084514, 18.21607675])
    ar_2022 = np.array(
        [16.42575, 16.67524, 17.3345, 17.74501, 17.50593, 17.4531, 16.60302, 16.63204, 16.77578, 16.83857, 16.2442,
         16.3977])
    ar_2023 = np.array(
        [15.64197, 16.34811, 17.14569, 17.48804, 17.97377, 17.67404, 16.24335, 16.20684, 16.18681, 16.34306, 15.73323,
         15.97414])
    ar_2024 = np.array(
        [14.34575, 14.8304, 15.40556, 15.9593, 16.18928, 15.54587, 13.87992, 13.86477, 13.88244, 13.76272, 13.31982,
         13.35429])

    actual = np.array([
        21.68, 21.68, 22.60, 22.71, 22.50, 20.75, 19.32, 18.18, 17.16, 16.75, 16.43, 16.34,  # 2021
        16.34, 16.34, 17.11, 17.36, 16.98, 16.60, 16.09, 16.34, 16.34, 16.21, 15.57, 15.43,  # 2022
        15.53, 16.10, 16.59, 16.97, 17.69, 17.43, 15.60, 15.43, 15.26, 15.29, 14.68647, 14.49,  # 2023
        14.34923, 14.7223, 15.17152, 15.78062, 15.94812, 14.91265, 12.99397, 12.48384, 12.41532, 12.43816, 12.2402, 12.26  # 2024
    ])
    actual_2021 = actual[:12]
    actual_2022 = actual[12:24]
    actual_2023 = actual[24:36]
    actual_2024 = actual[36:48]
    actual_train = np.concatenate([actual_2021, actual_2022, actual_2023])
    ar_train = np.concatenate([ar_2021, ar_2022, ar_2023])
    logging.info("Data loaded successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise

# Calculate errors for each month (2021–2023)
errors = {}
try:
    for year, ar, act in zip([2021, 2022, 2023], [ar_2021, ar_2022, ar_2023], [actual_2021, actual_2022, actual_2023]):
        errors[year] = ar - act
    logging.info("Errors calculated successfully")
except Exception as e:
    logging.error(f"Error calculating errors: {e}")
    raise

# Prepare training data for each month (1–12)
years = np.array([2021, 2022, 2023]).reshape(-1, 1)
ar_values = np.array([ar_2021, ar_2022, ar_2023])  # Shape: (3, 12)
months = np.array([list(range(1, 13))] * 3).T  # Shape: (12, 3)
X_2024 = np.array([[2024, ar_2024[i], i + 1] for i in range(12)])  # Year, AR value, month index

# Initialize models
models = {
    'GaussianProcess': None,  # Use 2023 error
    'Mean Error': None,
    'Prophet': None,
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=20, random_state=42, min_samples_split=2, min_samples_leaf=2),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=20, random_state=42, max_depth=2, min_samples_split=2, min_samples_leaf=2),
    'XGBoost': XGBRegressor(n_estimators=20, random_state=42, max_depth=2, min_child_weight=2),
    'SVR': SVR(kernel='rbf', C=0.5, epsilon=0.05)
}

# Calculate NSE
def calculate_nse(actual, pred):
    mean_actual = np.mean(actual)
    numerator = np.sum((actual - pred) ** 2)
    denominator = np.sum((actual - mean_actual) ** 2)
    nse = 1 - numerator / denominator if denominator != 0 else 0
    logging.debug(f"NSE calculation: mean_actual={mean_actual:.4f}, numerator={numerator:.4f}, denominator={denominator:.4f}, NSE={nse:.4f}")
    return nse

# Store corrected predictions and metrics
corrected_predictions = {}
corrected_predictions_train = {}  # For 2021–2023
metrics_2024 = {}
metrics_train = {}
yearly_models = {}  # Initialize globally

# Train and predict for each model
try:
    for name, model in models.items():
        corrected = np.zeros(12)  # For 2024
        corrected_train = np.zeros(36)  # For 2021–2023 (3 years x 12 months)
        logging.info(f"Processing model: {name}")

        if name in ['Linear Regression', 'Gradient Boosting', 'XGBoost']:
            # Train one model per year for Linear Regression, Gradient Boosting, and XGBoost
            for year in [2021, 2022, 2023]:
                # Prepare data for this year
                y_train = errors[year]  # 12 months of errors
                X_train = np.column_stack((np.full(12, year), ar_values[years.flatten().tolist().index(year), :], np.arange(1, 13)))  # Year, AR values, month indices
                current_model = models[name]  # Get the model instance
                current_model.fit(X_train, y_train)
                if year not in yearly_models:
                    yearly_models[year] = {}
                yearly_models[year][name] = current_model
                logging.debug(f"Trained {name} for year {year}")

            # Predict for 2024 using 2023 model
            for month in range(12):
                predicted_error_2024 = yearly_models[2023][name].predict(X_2024[month].reshape(1, -1))[0]
                corrected[month] = ar_2024[month] - predicted_error_2024
                logging.debug(f"Month {month + 1} (2024): predicted_error={predicted_error_2024:.4f}, corrected={corrected[month]:.4f}")

                # Predict for 2021–2023
                for i, year in enumerate([2021, 2022, 2023]):
                    idx = i * 12 + month
                    X_train_point = np.array([year, ar_values[i, month], month + 1]).reshape(1, -1)
                    predicted_error_train = yearly_models[year][name].predict(X_train_point)[0]
                    corrected_train[idx] = ar_values[i, month] - predicted_error_train
                    logging.debug(f"Year {year}, Month {month + 1}: predicted_error={predicted_error_train:.4f}, corrected={corrected_train[idx]:.4f}")

        else:
            # Original logic for other models
            for month in range(12):
                # Training data for this month
                y_train = np.array([errors[year][month] for year in [2021, 2022, 2023]])
                X_train = np.column_stack((years, ar_values[:, month], months[month]))  # Year, AR value, month index
                logging.debug(f"Month {month + 1}: y_train={y_train}, X_train={X_train}")
                if name == 'GaussianProcess':
                    predicted_error_2024 = errors[2023][month]  # Use 2023 error for 2024
                    predicted_errors_train = [errors[2023][month] for year in [2021, 2022, 2023]]
                elif name == 'Mean Error':
                    predicted_error_2024 = np.mean(y_train)
                    predicted_errors_train = [np.mean(y_train) for year in [2021, 2022, 2023]]
                elif name == 'Prophet':
                    predicted_error_2024 = np.median(y_train)
                    predicted_errors_train = [np.median(y_train) for year in [2021, 2022, 2023]]
                else:
                    model.fit(X_train, y_train)
                    predicted_error_2024 = model.predict(X_2024[month].reshape(1, -1))[0]
                    predicted_errors_train = [model.predict(np.array([year, ar_values[years.flatten().tolist().index(year), month], month + 1]).reshape(1, -1))[0] for year in [2021, 2022, 2023]]

                # 2024 predictions
                corrected[month] = ar_2024[month] - predicted_error_2024
                logging.debug(f"Month {month + 1} (2024): predicted_error={predicted_error_2024:.4f}, corrected={corrected[month]:.4f}")

                # 2021–2023 predictions
                for i, year in enumerate([2021, 2022, 2023]):
                    idx = i * 12 + month
                    corrected_train[idx] = ar_values[i, month] - predicted_errors_train[i]
                    logging.debug(f"Year {year}, Month {month + 1}: predicted_error={predicted_errors_train[i]:.4f}, corrected={corrected_train[idx]:.4f}")

        corrected_predictions[name] = corrected
        corrected_predictions_train[name] = corrected_train
        metrics_2024[name] = {
            'MAE': mean_absolute_error(actual_2024, corrected),
            'RMSE': np.sqrt(mean_squared_error(actual_2024, corrected)),
            'NSE': calculate_nse(actual_2024, corrected)
        }
    logging.info("Models trained and predictions made successfully")
except Exception as e:
    logging.error(f"Error in model training or prediction: {e}")
    raise

# Debug yearly_models contents
logging.debug("Yearly models for 2023: %s", {name: type(model) for name, model in yearly_models.get(2023, {}).items()})

# Original metrics for 2024
metrics_2024['Original'] = {
    'MAE': mean_absolute_error(actual_2024, ar_2024),
    'RMSE': np.sqrt(mean_squared_error(actual_2024, ar_2024)),
    'NSE': calculate_nse(actual_2024, ar_2024)
}

# Compute metrics for 2021–2023
model_names = ['Original', 'GaussianProcess', 'Mean Error', 'Prophet',
               'Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'SVR']
metrics_train['Original'] = {
    'MAE': mean_absolute_error(actual_train, ar_train),
    'RMSE': np.sqrt(mean_squared_error(actual_train, ar_train)),
    'NSE': calculate_nse(actual_train, ar_train)
}
for name in model_names[1:]:
    corrected = corrected_predictions_train[name]
    metrics_train[name] = {
        'MAE': mean_absolute_error(actual_train, corrected),
        'RMSE': np.sqrt(mean_squared_error(actual_train, corrected)),
        'NSE': calculate_nse(actual_train, corrected)
    }

# Print results
print("Evaluation Metrics for 2024:")
for name in model_names:
    print(f"\n{name}:")
    print(f"MAE: {metrics_2024[name]['MAE']:.4f}")
    print(f"RMSE: {metrics_2024[name]['RMSE']:.4f}")
    print(f"NSE: {metrics_2024[name]['NSE']:.4f}")

print("\nEvaluation Metrics for 2021–2023:")
for name in model_names:
    print(f"\n{name}:")
    print(f"MAE: {metrics_train[name]['MAE']:.4f}")
    print(f"RMSE: {metrics_train[name]['RMSE']:.4f}")
    print(f"NSE: {metrics_train[name]['NSE']:.4f}")

# Plotting predictions (2024 only)
try:
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 13), actual_2024, label='Actual', color='blue', linewidth=2, marker='o')
    plt.plot(range(1, 13), ar_2024, label='Original Prediction', color='red', linestyle='--', marker='x')
    colors = ['limegreen', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'gray']
    styles = ['-', '-.', ':', '-', '--', '-.', ':']
    for (name, corrected), color, style in zip(corrected_predictions.items(), colors, styles):
        plt.plot(range(1, 13), corrected, label=name, color=color, linestyle=style, marker='s')
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title('2024 Predictions: Actual vs Original vs Corrected (Month-Specific Correction)')
    plt.ylim(0, None)  # Set Y-axis to start at 0
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot.png')  # Save plot
    plt.show()
    logging.info("Prediction plot generated and saved as plot.png")
except Exception as e:
    logging.error(f"Error in plotting predictions: {e}")
    raise

# Plotting combined predictions (2021–2023 and 2024)
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1]}, sharey=False)

    # Left subplot: 2021–2023
    train_years = np.arange(36)  # 36 months (Jan 2021–Dec 2023)
    ax1.plot(train_years, actual_train, label='Actual', color='blue', linewidth=2, marker='o')
    ax1.plot(train_years, ar_train, label='Original Prediction', color='red', linestyle='--', marker='x')
    for (name, corrected), color, style in zip(corrected_predictions_train.items(), colors, styles):
        ax1.plot(train_years, corrected, label=name, color=color, linestyle=style, marker='s')

    # Compute Y-axis limits for left subplot
    all_train_values = np.concatenate([actual_train, ar_train] + [corr for corr in corrected_predictions_train.values()])
    min_train = np.min(all_train_values)
    max_train = np.max(all_train_values)
    margin_train = (max_train - min_train) * 0.05  # 5% margin
    ax1.set_ylim(min_train - margin_train, max_train + margin_train)

    ax1.set_xlabel('Date (2021–2023)')
    ax1.set_ylabel('Value')
    ax1.set_title('2021–2023 Predictions: Actual vs Original vs Corrected')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(np.arange(0, 36, 12))
    ax1.set_xticklabels(['2021', '2022', '2023'])

    # Right subplot: 2024
    ax2.plot(range(1, 13), actual_2024, label='Actual', color='blue', linewidth=2, marker='o')
    ax2.plot(range(1, 13), ar_2024, label='Original Prediction', color='red', linestyle='--', marker='x')
    for (name, corrected), color, style in zip(corrected_predictions.items(), colors, styles):
        ax2.plot(range(1, 13), corrected, label=name, color=color, linestyle=style, marker='s')

    # Compute Y-axis limits for right subplot
    all_2024_values = np.concatenate([actual_2024, ar_2024] + [corr for corr in corrected_predictions.values()])
    min_2024 = np.min(all_2024_values)
    max_2024 = np.max(all_2024_values)
    margin_2024 = (max_2024 - min_2024) * 0.05  # 5% margin
    ax2.set_ylim(min_2024 - margin_2024, max_2024 + margin_2024)

    ax2.set_xlabel('Month (2024)')
    ax2.set_ylabel('Value')
    ax2.set_title('2024 Predictions: Actual vs Original vs Corrected')
    ax2.grid(True)
    ax2.legend()

    # Log Y-axis limits for verification
    logging.debug(f"Left subplot Y-axis limits: {ax1.get_ylim()}")
    logging.debug(f"Right subplot Y-axis limits: {ax2.get_ylim()}")

    plt.tight_layout()
    plt.savefig('combined_predictions.png')
    plt.show()
    logging.info("Combined predictions plot generated and saved as combined_predictions.png")
except Exception as e:
    logging.error(f"Error in plotting combined predictions: {e}")
    raise

# Prepare radar chart data
def plot_radar_chart(values, title, filename):
    num_vars = len(model_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=100)

    # Color palette (modern and professional)
    colors = sns.color_palette("husl", 3)  # Use HSL-based palette for distinct colors
    line_styles = [':', ':', ':']  # Dotted lines for MAE, RMSE, NSE
    alpha_fills = [0.15, 0.1, 0.2]  # Subtle fill transparency

    # Plot each metric
    for i, (metric_name, vals) in enumerate(zip(['MAE', 'RMSE', 'NSE'], values)):
        data = np.concatenate([vals, [vals[0]]])  # Close the loop
        line, = ax.plot(angles, data, label=metric_name, color=colors[i], linewidth=2.5,
                        linestyle=line_styles[i], marker='o', markersize=2)
        ax.fill(angles, data, color=colors[i], alpha=alpha_fills[i])

        # Add annotations only for NSE
        if metric_name == 'NSE':
            for j, (angle, val) in enumerate(zip(angles[:-1], vals)):
                if abs(val) > 0.1:  # Annotate significant values to avoid clutter
                    ax.text(angle, val + 0.05 * abs(val), f'{val:.3f}', color=colors[i],
                            fontsize=10, ha='center', va='center')

    # Set Y-axis limits with margin
    all_values = np.concatenate(values)
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1  # Avoid zero margin
    ax.set_ylim(min_val - margin, max_val + margin)

    # Draw custom solid outer circle
    ax.spines['polar'].set_visible(False)  # Disable default spine
    ax.plot(angles, [max_val + margin] * len(angles), color='black', linestyle='solid', linewidth=1.5)

    # Customize axes with fixed number of ticks for consistency
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_names, fontsize=14, weight='medium')
    ax.set_rlabel_position(0)
    yticks = np.linspace(min_val - margin, max_val + margin, 5)  # Fixed 5 ticks
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in yticks], fontsize=12)
    ax.yaxis.set_tick_params(pad=10)  # Move radial labels outward

    # Customize grid and background
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_facecolor('#ffffff')  # White background for axes
    fig.patch.set_facecolor('#ffffff')  # White background for figure

    # Rotate and align model labels
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle > np.pi / 2 and angle < 3 * np.pi / 2:
            label.set_horizontalalignment('right')
            label.set_rotation(180 * angle / np.pi + 90)
        else:
            label.set_horizontalalignment('left')
            label.set_rotation(180 * angle / np.pi - 90)

    # Add title and legend
    plt.title(title, size=16, weight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15), fontsize=12,
              frameon=True, edgecolor='black', shadow=True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    logging.info(f"Radar chart generated and saved as {filename}")

# Radar chart for 2024
mae_values_2024 = np.array([metrics_2024[name]['MAE'] for name in model_names])
rmse_values_2024 = np.array([metrics_2024[name]['RMSE'] for name in model_names])
nse_values_2024 = np.array([metrics_2024[name]['NSE'] for name in model_names])
values_2024 = np.array([mae_values_2024, rmse_values_2024, nse_values_2024])

logging.info("Raw metrics for 2024 radar chart: " + ", ".join(
    [f"{name}: MAE={mae_values_2024[i]:.4f}, RMSE={rmse_values_2024[i]:.4f}, NSE={nse_values_2024[i]:.4f}"
     for i, name in enumerate(model_names)]))

plot_radar_chart(values_2024, 'Model Performance Comparison(2024)', 'metrics_comparison_2024_raw_advanced.png')

# Radar chart for 2021–2023
mae_values_train = np.array([metrics_train[name]['MAE'] for name in model_names])
rmse_values_train = np.array([metrics_train[name]['RMSE'] for name in model_names])
nse_values_train = np.array([metrics_train[name]['NSE'] for name in model_names])
values_train = np.array([mae_values_train, rmse_values_train, nse_values_train])

logging.info("Raw metrics for 2021–2023 radar chart: " + ", ".join(
    [f"{name}: MAE={mae_values_train[i]:.4f}, RMSE={rmse_values_train[i]:.4f}, NSE={nse_values_train[i]:.4f}"
     for i, name in enumerate(model_names)]))

plot_radar_chart(values_train, 'Model Performance Comparison(2021–2023)', 'metrics_comparison_2021_2023_raw_advanced.png')

# 整理数据
output_data = {
    'Month': list(range(1, 13)),
    'Actual': actual_2024,
    'Original Prediction': ar_2024
}
for name, preds in corrected_predictions.items():
    output_data[name] = preds

# 转换为DataFrame
df_output = pd.DataFrame(output_data)

# 输出到 Excel 文件
output_excel_path = '2024_predictions_comparison.xlsx'
df_output.to_excel(output_excel_path, index=False)

logging.info(f"2024 prediction results saved to Excel: {output_excel_path}")
print(f"✅ Excel 文件已保存：{output_excel_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


# from your_module import models, yearly_models, errors, years, ar_values, months

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
# from your_module import models, yearly_models, errors, years, ar_values, months

def process_scenario_data(sheet_name):
    # ... (前段数据读取和处理保持不变) ...
    # Read scenario data
    try:
        scenario_df = pd.read_excel('9种情景设计结果.xlsx', engine='openpyxl', sheet_name=sheet_name)
        logging.info(f"Scenario data loaded successfully from sheet: {sheet_name}")

        column_mapping = {
            '情景': 'Scenario',
            '月份': 'Month',
            '预测值 (米)': 'Forecast',
            '下限 (米)': 'Lower Bound',
            '上限 (米)': 'Upper Bound'
        }
        scenario_df = scenario_df.rename(columns=column_mapping)

        required_columns = ['Scenario', 'Month', 'Forecast']
        if not all(col in scenario_df.columns for col in required_columns):
            raise KeyError(f"Excel file must contain columns: {required_columns}")

        scenario_dfs = {}
        for i in range(1, 10):
            scenario_name = f'Scenario{i}'
            scenario_data = scenario_df[scenario_df['Scenario'] == f'情景{i}'].copy()
            if not scenario_data.empty:
                scenario_data['Actual'] = None
                scenario_data['Error'] = None
                scenario_dfs[scenario_name] = scenario_data

        if not scenario_dfs:
            raise ValueError("No valid scenario data found in Excel file")

    except Exception as e:
        logging.error(f"Error reading scenario Excel file: {e}")
        raise

    amplification_factors = {f'Scenario{i}': 1 for i in range(1, 10)}

    corrected_scenario_predictions = {}
    try:
        for scenario_name, scenario_data in scenario_dfs.items():
            corrected_predictions_scenario = {}
            scenario_forecasts = scenario_data['Forecast'].values
            X_scenario = np.array([[2024, scenario_forecasts[i], i + 1] for i in range(12)])

            for name, model in models.items():
                corrected = np.zeros(12)
                if name in ['Linear Regression', 'Gradient Boosting', 'XGBoost']:
                    if 'yearly_models' not in globals() or 2023 not in yearly_models or name not in yearly_models[2023]:
                        continue
                    yr_model = yearly_models[2023][name]
                    for month in range(12):
                        predicted_error = yr_model.predict(X_scenario[month].reshape(1, -1))[0] * amplification_factors[
                            scenario_name]
                        corrected[month] = scenario_forecasts[month] - predicted_error
                else:
                    for month in range(12):
                        y_train = np.array([errors[year][month] for year in [2021, 2022, 2023]])
                        if name == 'GaussianProcess':
                            predicted_error = errors[2023][month] * amplification_factors[scenario_name]
                        elif name == 'Mean Error':
                            predicted_error = np.mean(y_train) * amplification_factors[scenario_name]
                        elif name == 'Prophet':
                            predicted_error = np.median(y_train) * amplification_factors[scenario_name]
                        else:
                            X_train = np.column_stack((years, ar_values[:, month], months[month]))
                            model.fit(X_train, y_train)
                            predicted_error = model.predict(X_scenario[month].reshape(1, -1))[0] * \
                                              amplification_factors[scenario_name]

                        corrected[month] = scenario_forecasts[month] - predicted_error

                corrected_scenario_predictions[scenario_name] = corrected_predictions_scenario
                corrected_predictions_scenario[name] = corrected

            corrected_scenario_predictions[scenario_name] = corrected_predictions_scenario

    except Exception as e:
        logging.error(f"Error in scenario correction: {e}")
        raise

    # ---------------------------------------------------------
    # Visualization: Large Fonts & Wide Layout
    # ---------------------------------------------------------
    try:
        # 全局字体参数 (保持大字体)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 24,
            'axes.linewidth': 2.0,
            'grid.linewidth': 1.0,
            'legend.fontsize': 22,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20
        })

        # 计算Y轴范围
        all_forecasts = np.concatenate([df['Forecast'].values for df in scenario_dfs.values()])
        all_corrected = np.concatenate([
            corrected_scenario_predictions[scenario][name]
            for scenario in scenario_dfs
            for name in models
        ])

        has_bounds = 'Lower Bound' in list(scenario_dfs.values())[0].columns and \
                     'Upper Bound' in list(scenario_dfs.values())[0].columns

        y_min_candidates = [all_forecasts.min(), all_corrected.min(), 11.0]
        y_max_candidates = [all_forecasts.max(), all_corrected.max(), 13.0]

        if has_bounds:
            all_lower = np.concatenate([df['Lower Bound'].values for df in scenario_dfs.values()])
            all_upper = np.concatenate([df['Upper Bound'].values for df in scenario_dfs.values()])
            y_min_candidates.append(all_lower.min())
            y_max_candidates.append(all_upper.max())

        y_min = min(y_min_candidates) - 0.5
        y_max = max(y_max_candidates) + 0.5

        # 画布尺寸 (保持宽大)
        fig, axes = plt.subplots(3, 3, figsize=(24, 20), dpi=300)
        axes = axes.flatten()

        main_model_name = 'Mean Error'

        model_colors = {
            'Mean Error': '#0044AA',
            'GaussianProcess': '#008000',
            'Prophet': '#800080',
            'Linear Regression': '#8B4513',
            'Random Forest': '#C71585',
            'Gradient Boosting': '#696969',
            'XGBoost': '#808000',
            'SVR': '#008B8B'
        }

        for idx, (scenario_name, scenario_data) in enumerate(scenario_dfs.items()):
            ax = axes[idx]
            scenario_data = scenario_data.sort_values('Month')
            months_axis = range(1, 13)

            # 风险区域（斜线阴影）
            ax.fill_between(months_axis, 0, 11.7,
                            facecolor='none',
                            hatch='///',
                            edgecolor='#333333',
                            alpha=0.5,
                            zorder=0, linewidth=0)

            if has_bounds:
                ax.fill_between(scenario_data['Month'],
                                scenario_data['Lower Bound'],
                                scenario_data['Upper Bound'],
                                color='gray', alpha=0.15, zorder=1, linewidth=0)

            for name in models:
                if name == main_model_name: continue
                corrected = corrected_scenario_predictions[scenario_name][name]
                if np.all(corrected == 0) or np.any(np.isnan(corrected)): continue
                color = model_colors.get(name, 'gray')
                ax.plot(scenario_data['Month'], corrected, color=color, linestyle='-',
                        linewidth=2.0, alpha=0.75, zorder=2)

            # 原始预测
            ax.plot(scenario_data['Month'], scenario_data['Forecast'],
                    color='black', linewidth=2.5, linestyle='--',
                    marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2.0,
                    zorder=3, alpha=1.0)

            ax.axhline(y=11.7, color='#D62728', linestyle='-', linewidth=2.0, alpha=1.0, zorder=1)
            ax.axhline(y=12.0, color='#FF7F0E', linestyle=':', linewidth=2.0, alpha=1.0, zorder=1)

            if main_model_name in corrected_scenario_predictions[scenario_name]:
                corrected = corrected_scenario_predictions[scenario_name][main_model_name]
                ax.plot(scenario_data['Month'], corrected,
                        color=model_colors[main_model_name], linestyle='-', linewidth=4.0,
                        marker='D', markersize=8, zorder=10)

            # 标题和标签
            ax.set_title(f'{scenario_name}', fontsize=28, fontweight='bold', pad=15, color='black')
            if idx % 3 == 0:
                ax.set_ylabel('Groundwater Depth (m)', fontsize=24, labelpad=10, fontweight='bold')
            if idx >= 6:
                ax.set_xlabel('Month', fontsize=24, labelpad=10, fontweight='bold')

            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0.8, 12.2)
            ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.4, color='#666666')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(colors='black', width=1.5, length=6)
            ax.set_xticks(range(1, 13))

        # --- 图例设置 (关键修改部分) ---
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_main = [
            Line2D([0], [0], color=model_colors[main_model_name], lw=4.0, marker='D', markersize=8,
                   label=f'Corrected ({main_model_name})'),
            Line2D([0], [0], color='black', lw=2.5, linestyle='--', marker='o', markersize=7, markerfacecolor='white',
                   label='Original Prediction'),
            Line2D([0], [0], color='#D62728', lw=2.0, linestyle='-', label='Critical (11.7m)'),
            Line2D([0], [0], color='#FF7F0E', lw=2.0, linestyle=':', label='Alert (12.0m)'),
            Patch(facecolor='white', hatch='///', edgecolor='#333333', alpha=0.5, label='Risk Zone (<11.7m)')
        ]

        if has_bounds:
            legend_main.append(Patch(facecolor='gray', alpha=0.4, label='Prediction Range'))

        legend_ref = []
        other_models = sorted([m for m in models if m != main_model_name])
        for name in other_models:
            color = model_colors.get(name, 'gray')
            legend_ref.append(Line2D([0], [0], color=color, lw=2.5, alpha=1.0, label=name))

        # 【关键修改 1】 主图例：ncol设为6，强制排成一行
        # columnspacing设为1.2，利用宽屏幕优势拉开一点
        leg1 = fig.legend(handles=legend_main, loc='lower center', bbox_to_anchor=(0.5, 0.08),
                          ncol=6, frameon=False, fontsize=22, columnspacing=1.2)

        # 【关键修改 2】 参考模型图例：ncol设为8，强制排成一行
        # 这里的模型数量大约是7个左右，ncol=8足够把它们全部放进一行
        leg2 = fig.legend(handles=legend_ref, loc='lower center', bbox_to_anchor=(0.5, 0.03),
                          ncol=8, frameon=False, fontsize=20, columnspacing=1.0, handletextpad=0.4)

        fig.add_artist(leg1)

        # 稍微减少一点底部边距，因为图例现在扁了（只有两行总高度）
        plt.subplots_adjust(top=0.95, bottom=0.18, left=0.10, right=0.95, hspace=0.4, wspace=0.25)

        save_name = f'scenario_correction_large_font_{sheet_name}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        logging.info(f"Large font scenario plot saved as {save_name}")

    except Exception as e:
        logging.error(f"Error in scenario visualization: {e}")
        raise

# Example usage: replace 'Sheet1' with your actual sheet name
process_scenario_data('30')
process_scenario_data('35')
process_scenario_data('40')
