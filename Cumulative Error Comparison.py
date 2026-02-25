import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "Autoregressive Forecast": [
        16.49755031, 16.74703351, 17.40629809, 17.8168076, 17.57772733, 17.52489273,
        16.6748147, 16.70383694, 16.84757194, 16.9103679, 16.31599295, 16.46949356,
        16.77031833, 17.47645316, 18.27403308, 18.61638922, 19.10212164, 18.80238315,
        17.37169877, 17.33518514, 17.31515665, 17.47140468, 16.86157732, 17.10248695,
        16.9604773, 17.44512397, 18.02028779, 18.57402321, 18.80400901, 18.16059713,
        16.49464698, 16.479496, 16.49717172, 16.37745157, 15.93454383, 15.96902103
    ],
    "Rolling Forecast": [
        16.49755031, 16.5894832, 16.99926458, 17.52050951, 17.12091973, 16.9271654,
        15.74992197, 16.11902224, 16.483735, 16.40279596, 15.61562505, 15.72350062,
        15.72850476, 16.23170483, 16.89653992, 16.90282614, 17.45177241, 17.39352152,
        15.99685562, 15.55898637, 15.4076515, 15.41610803, 14.67800264, 14.92737963,
        14.35188035, 14.83387667, 15.29746382, 15.72525543, 16.0106058, 15.30470812,
        13.24669985, 12.97881902, 12.50151572, 12.29559985, 11.99525226, 12.2746772
    ],
    "Observed": [
        16.34, 16.34, 17.11, 17.36, 16.98, 16.60,
        16.09, 16.34, 16.34, 16.21, 15.57, 15.43,
        15.53, 16.10, 16.56, 16.97, 17.69, 17.43,
        15.60, 15.43, 15.26, 15.29, 14.68647, 14.79,
        14.34923, 14.7223, 15.17152, 15.78062, 15.94812, 14.91265,
        12.99397, 12.48384, 12.41532, 12.43816, 12.2402, 12.26
    ]
}

df = pd.DataFrame(data)

# Time axis (monthly)
df["Date"] = pd.date_range(start="2021-12-01", periods=len(df), freq='M').strftime("%Y-%m")

# Plotting
plt.figure(figsize=(12, 6))

# Subplot 1: Rolling Forecast vs Observed
plt.subplot(2, 1, 1)
plt.plot(df["Date"], df["Rolling Forecast"], label='Rolling Forecast', color='orange', marker='o')
plt.plot(df["Date"], df["Observed"], label='Observed', color='blue', marker='o')
plt.title('Rolling Forecast Results')
plt.ylabel('Depth (m)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Subplot 2: Autoregressive Forecast vs Observed
plt.subplot(2, 1, 2)
plt.plot(df["Date"], df["Autoregressive Forecast"], label='Autoregressive Forecast', color='orange', marker='o')
plt.plot(df["Date"], df["Observed"], label='Observed', color='blue', marker='o')
plt.title('Autoregressive Forecast Results')
plt.ylabel('Depth (m)')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
