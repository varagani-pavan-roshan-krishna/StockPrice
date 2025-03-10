from flask import Flask, render_template, request, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained model
MODEL_PATH = "stock_dl_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

model = load_model(MODEL_PATH)


# Function to fetch stock data
def fetch_stock_data(stock_symbol, start="2000-01-01", end="2024-11-01"):
    try:
        stock_data = yf.download(stock_symbol, start=start, end=end)
        if stock_data.empty:
            raise ValueError("No data retrieved for the stock.")
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


# Function to generate EMA plots
def plot_ema(stock_data, short_window, long_window, filename):
    stock_data[f'EMA_{short_window}'] = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
    stock_data[f'EMA_{long_window}'] = stock_data['Close'].ewm(span=long_window, adjust=False).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label="Closing Price", color="blue")
    plt.plot(stock_data[f'EMA_{short_window}'], label=f"EMA {short_window}", color="red")
    plt.plot(stock_data[f'EMA_{long_window}'], label=f"EMA {long_window}", color="green")
    plt.legend()
    plt.title(f"Stock Price & EMA ({short_window}-{long_window})")
    plt.savefig(f"static/{filename}")
    plt.close()


# Function to prepare data for prediction
def prepare_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    X_test = []
    for i in range(100, len(scaled_data)):
        X_test.append(scaled_data[i - 100:i, 0])

    return np.array(X_test).reshape(-1, 100, 1), scaler


# Function to make predictions
def make_prediction(stock_data):
    X_test, scaler = prepare_data(stock_data)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions


@app.route("/", methods=["GET", "POST"])
def index():
    plot_path_ema_20_50, plot_path_ema_100_200, plot_path_prediction, data_desc, dataset_link, error_message = None, None, None, None, None, None

    if request.method == "POST":
        stock_symbol = request.form.get("stock", "POWERGRID.NS")
        stock_data = fetch_stock_data(stock_symbol)

        if stock_data is not None:
            # Save dataset
            dataset_path = f"static/{stock_symbol}.csv"
            stock_data.to_csv(dataset_path)
            dataset_link = dataset_path

            # Generate EMA plots
            plot_ema(stock_data, 20, 50, "ema_20_50.png")
            plot_ema(stock_data, 100, 200, "ema_100_200.png")
            plot_path_ema_20_50 = "static/ema_20_50.png"
            plot_path_ema_100_200 = "static/ema_100_200.png"

            # Make predictions
            predictions = make_prediction(stock_data)
            stock_data = stock_data.iloc[-len(predictions):]
            stock_data["Predicted"] = predictions

            plt.figure(figsize=(10, 5))
            plt.plot(stock_data["Close"], label="Actual Price", color="blue")
            plt.plot(stock_data["Predicted"], label="Predicted Price", color="red")
            plt.legend()
            plt.title("Stock Price Prediction")
            plt.savefig("static/stock_prediction.png")
            plt.close()
            plot_path_prediction = "static/stock_prediction.png"

            # Descriptive statistics
            data_desc = stock_data.describe().to_html()
        else:
            error_message = "Invalid stock symbol or data unavailable for the given period."

    return render_template("index.html", plot_path_ema_20_50=plot_path_ema_20_50,
                           plot_path_ema_100_200=plot_path_ema_100_200,
                           plot_path_prediction=plot_path_prediction,
                           data_desc=data_desc, dataset_link=dataset_link,
                           error_message=error_message)


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
