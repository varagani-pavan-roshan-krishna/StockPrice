Stock Price Prediction

Overview

This project is a web-based stock price prediction application built using Flask, Keras, and Yahoo Finance data. The application fetches stock data, processes it, and predicts future stock prices using a deep learning model.

Features

Fetch real-time stock data from Yahoo Finance

Perform data preprocessing and normalization

Predict future stock prices using a trained deep learning model

Generate and display technical analysis charts including Exponential Moving Averages (EMA)

Download stock data as a CSV file

Technologies Used

Python: Data processing and model loading

Flask: Web framework for building the application

Keras: Deep learning model for stock price prediction

Yahoo Finance API (yfinance): Fetching stock data

Pandas & NumPy: Data processing

Matplotlib: Visualization of stock trends

Scikit-learn (MinMaxScaler): Data normalization

Installation

Clone the repository:

git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Usage

Open the application in a web browser: http://127.0.0.1:5000/

Enter a stock ticker symbol (e.g., AAPL, GOOGL) and submit.

View predictions, stock trends, and download the dataset.

File Structure

app.py: Main Flask application

stock_dl_model.h5: Pre-trained deep learning model

templates/index.html: Frontend template

static/: Directory for generated images and CSV files

Screenshots

# Stock Analysis

## Exponential Moving Averages (EMA)

### EMA 20 & 50
![EMA 20 & 50](static/ema_20_50.png)

### EMA 100 & 200
![EMA 100 & 200](static/ema_100_200.png)

## Stock Prediction
![Stock Prediction](static/stock_prediction.png)


License

This project is open-source and available under the MIT License.

Author

Developed by Pavan Roshan.

Contact

For queries, reach out to: vprk.2468@gmail.com