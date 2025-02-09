# Price Prediction Using LSTM and Data Processing

## Overview

This project focuses on extracting price data from HTML files, processing it into structured data, and predicting future prices using an LSTM (Long Short-Term Memory) neural network.

## Features

- **Data Extraction**: Parses HTML tables using BeautifulSoup and converts them into Pandas DataFrames.
- **Data Processing**: Cleans, structures, and organizes data into meaningful columns.
- **Data Splitting**: Divides data into training and validation sets.
- **LSTM Model**: Implements an LSTM-based neural network for price prediction.
- **Visualization**: Plots actual vs. predicted prices for better insights.

## Requirements

Ensure you have the following libraries installed before running the script:

```sh
pip install pandas beautifulsoup4 numpy tensorflow matplotlib
```

## Necessary Imports

```python
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import datetime
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
```

## Data Extraction

The `upload` function extracts and processes price data from HTML tables.

```python
def upload(your_file, date, k):
    with open(your_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    df = pd.read_html(StringIO(str(tables[0])))[0].iloc[5:10]
    df.reset_index(drop=True, inplace=True)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True).drop(df.columns[0], axis=1)
    new_columns = [f'{col}_{i+1}' for i in range(df.shape[0]) for col in df.columns]
    new_row = [item for sublist in df.values.tolist() for item in sublist]
    new_df = pd.DataFrame([new_row], columns=new_columns, index=[date])
    return new_df
```

## Data Processing

The `create` function generates DataFrames for a given date range.

```python
def create(start_date, end_date, k):
    date_list = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    dataframes = {f'df{i}': upload(f'{i}.html', date.strftime('%d/%m/%Y'), k) for i, date in enumerate(date_list)}
    df_list = [df for df in dataframes.values()]
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.index = date_list
    return final_df
```

## Training the LSTM Model

```python
model = Sequential([
    InputLayer((5, 1)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
```

## Making Predictions

```python
recent_prices = X1[-1].reshape((1, 5, 1))
predictions = []
for _ in range(5):
    next_value = model.predict(recent_prices)[0, 0]
    predictions.append(next_value)
    recent_prices = np.roll(recent_prices, shift=-1, axis=1)
    recent_prices[0, -1, 0] = next_value
print("Predicted prices for the next 5 days:", predictions)
```

## Visualization

```python
plt.figure(figsize=(10, 5))
plt.plot(tdf.index[tdf.index < m_i], tdf[price][tdf.index < m_i], color='darkblue', marker='o', label='Actual Price')
plt.plot(tdf.index[tdf.index >= m_i], tdf[price][tdf.index >= m_i], color='lightblue', marker='x', linestyle='--', label='Predicted Price')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
```

## Conclusion

This project successfully extracts and processes price data from HTML files, trains an LSTM model, and predicts future prices. The results are visualized for analysis and decision-making.

