# Data Visualization Web App

A Flask web application for visualizing data analysis results with interactive plots.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have the data file `concatenated_results_2025-09-06.csv` in the project root directory
2. Run the Flask application:
   ```
   python app.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`

## Features

The application provides three types of visualizations:
- Ridge plots (accessible at `/ridge_data`)
- Horizontal bar plots (accessible at `/bar_plot`) 
- Heatmap plots (accessible at `/heatmap_plot`)

## Project Structure

- `app.py` - Main Flask application
- `plot_generator.py` - Contains plotting functions
- `templates/index.html` - Main web interface
- `static/style.css` - Styling for the web interface
- `concatenated_results_2025-09-06.csv` - Data file (required)

## Requirements

See `requirements.txt` for a complete list of Python dependencies.
