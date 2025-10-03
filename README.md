# Demand-Forecasting-on-Blinkit-Sales-Dataset
This project analyzes historical sales data to forecast future demand by product categories using Prophet, helping businesses plan inventory efficiently. It also generates visualizations and a combined CSV of predicted monthly sales for easy reference.
Step 1: Data Loading

I loaded the sales dataset into Python using pandas and inspected it to understand the structure, including columns like order_date, product_name, category, and quantity.

Step 2: Data Cleaning

I ensured the order_date column was in datetime format and checked for missing or inconsistent values. This made sure the data was ready for accurate aggregation and forecasting.

Step 3: Data Aggregation

I aggregated the sales data monthly by category, summing the quantities sold. This step converted transactional data into a time-series format suitable for forecasting.

Step 4: Preparing for Forecasting

I listed all unique categories and prepared the dataset for each category separately, keeping only the necessary columns (ds for date and y for quantity) required by Prophet.

Step 5: Forecasting with Prophet

I applied Facebook’s Prophet model to each category to predict future demand for the next 6 months. For categories with insufficient historical data, I skipped forecasting to maintain accuracy.

Step 6: Saving Forecast Results

For each category, I saved:

A plot of the forecast as a PNG file for visual analysis.

All forecasts were combined into a single CSV file containing predicted values and confidence intervals.

Step 7: Visualization and Reporting

I generated visualizations to observe seasonal trends and demand patterns, which help in inventory planning, stock management, and decision-making for different product categories.
