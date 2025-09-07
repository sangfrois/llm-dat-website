from flask import Flask, render_template, jsonify
import pandas as pd
import json
import plotly
from plot_generator import create_ridge_data, create_horizontal_bar_plot, create_heatmap_plot

app = Flask(__name__)

# Load data once at startup
results_df = pd.read_csv('concatenated_results_2025-09-06.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ridge_data')
def ridge_data():
    data = create_ridge_data(results_df)
    return jsonify(data)

@app.route('/bar_plot')
def bar_plot():
    fig = create_horizontal_bar_plot(results_df)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/heatmap_plot')
def heatmap_plot():
    fig = create_heatmap_plot(results_df)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
