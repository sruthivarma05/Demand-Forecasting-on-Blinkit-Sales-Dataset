# Blinkit Data Cleaning & Merging

import os
import pandas as pd

# Set Working Directory 
os.chdir(r"C:\Users\sruthi\Pictures\Downloads\New folder (6)")

# Loading Datasets 
orders = pd.read_csv("blinkit_orders.csv")
order_items = pd.read_csv("blinkit_order_items.csv")
products = pd.read_csv("blinkit_products.csv")
customers = pd.read_csv("blinkit_customers.csv")
delivery = pd.read_csv("blinkit_delivery_performance.csv")
feedback = pd.read_csv("blinkit_customer_feedback.csv")

# Data Cleaning 

# Converting dates
if 'order_date' in orders.columns:
    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')

# Droping rows with missing IDs
orders.dropna(subset=[col for col in ['order_id', 'customer_id'] if col in orders.columns], inplace=True)

# Filling missing numeric values
if 'price' in products.columns:
    products['price'].fillna(products['price'].median(), inplace=True)
if 'age' in customers.columns:
    customers['age'].fillna(customers['age'].median(), inplace=True)
if 'delivery_time' in delivery.columns:
    delivery['delivery_time'].fillna(delivery['delivery_time'].median(), inplace=True)
if 'rating' in feedback.columns:
    feedback['rating'].fillna(feedback['rating'].median(), inplace=True)

# Filling missing categorical values safely
for col in ['category', 'city']:
    if col in products.columns:
        products[col].fillna("Unknown", inplace=True)
        products[col] = products[col].str.strip().str.title()
    if col in customers.columns:
        customers[col].fillna("Unknown", inplace=True)
        customers[col] = customers[col].str.strip().str.title()

# Removing duplicates
orders.drop_duplicates(inplace=True)
order_items.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
customers.drop_duplicates(inplace=True)

# Merging Datasets 

# Sales Data: Orders + Order Items + Products
sales_data = orders.merge(order_items, on="order_id", how="inner")
if 'product_id' in sales_data.columns and 'product_id' in products.columns:
    sales_data = sales_data.merge(products, on="product_id", how="inner")

# Customer Orders: Orders + Customers
customer_orders = orders.copy()
if 'customer_id' in orders.columns and 'customer_id' in customers.columns:
    customer_orders = orders.merge(customers, on="customer_id", how="inner")

# Delivery Analysis: Orders + Delivery Performance
delivery_analysis = orders.copy()
if 'order_id' in orders.columns and 'order_id' in delivery.columns:
    delivery_analysis = orders.merge(delivery, on="order_id", how="inner")

# Feedback Analysis: Orders + Feedback
feedback_data = orders.copy()
if 'order_id' in orders.columns and 'order_id' in feedback.columns:
    feedback_data = orders.merge(feedback, on="order_id", how="inner")

# Saving Merged Datasets 
sales_data.to_csv("merged_sales_data.csv", index=False)
customer_orders.to_csv("merged_customer_orders.csv", index=False)
delivery_analysis.to_csv("merged_delivery_analysis.csv", index=False)
feedback_data.to_csv("merged_feedback_data.csv", index=False)

# Data Cleaning Check

datasets = {
    "Orders": orders,
    "Order Items": order_items,
    "Products": products,
    "Customers": customers,
    "Delivery": delivery,
    "Feedback": feedback
}

for name, df in datasets.items():
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Columns:")
    print(df.columns.tolist())

#cheking dataset columns for demad forecasting
print(sales_data.columns)

# Checking first few rows
print(sales_data[['order_date','product_name','quantity']].head())

# Checking for missing values
print(sales_data[['order_date','product_name','quantity']].isnull().sum())


# Ensuring order_date is datetime
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])

# Aggregating daily demand per product
daily_demand = sales_data.groupby(['order_date','product_name'])['quantity'].sum().reset_index()

# Aggregating monthly demand
monthly_demand = daily_demand.copy()
monthly_demand['month'] = monthly_demand['order_date'].dt.to_period('M')
monthly_demand = monthly_demand.groupby(['month','product_name'])['quantity'].sum().reset_index()

# Preview
print(daily_demand.head())
print(monthly_demand.head())

# ===============================
# Step 6: Demand Forecasting by Category
# ===============================

from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Ensure order_date is datetime
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])

# Aggregate monthly demand per category
monthly_demand = sales_data.groupby(['category', pd.Grouper(key='order_date', freq='M')])['quantity'].sum().reset_index()
monthly_demand.rename(columns={'order_date':'ds','quantity':'y'}, inplace=True)

# List of unique categories
categories = monthly_demand['category'].unique()

# Forecast horizon (months)
forecast_periods = 6

# List to store all forecasts for combined CSV
all_forecasts = []

for category in categories:
    print(f"\nForecasting demand for category: {category}")
    
    # Prepare data for Prophet
    category_data = monthly_demand[monthly_demand['category'] == category][['ds','y']]
    
    # Skip categories with too few data points
    if len(category_data) < 6:
        print(f"  Skipped {category} (not enough data)")
        continue
    
    # Initialize and fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(category_data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
    forecast = model.predict(future)
    
    # Add category column for combined CSV
    forecast['category'] = category
    all_forecasts.append(forecast[['category','ds','yhat','yhat_lower','yhat_upper']])
    
    # Save forecast plot automatically
    fig = model.plot(forecast)
    plt.title(f"Forecasted Demand for Category: {category}")
    plt.xlabel("Month")
    plt.ylabel("Quantity Sold")
    plt.savefig(f"forecast_plot_{category.replace(' ','_')}.png")  # save plot
    plt.close()

# Combine all forecasts into a single CSV
combined_forecast = pd.concat(all_forecasts)
combined_forecast.to_csv("combined_category_forecast.csv", index=False)

print("\n✅ Forecasting complete. All forecasts saved to combined CSV and plots saved as PNGs.")


