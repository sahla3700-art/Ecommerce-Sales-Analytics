"""
E-Commerce Sales and Customer Analytics using AI
IBM SkillsBuild Data Analytics with AI Academic Internship

Author: Sahla Fathima P S
Project: E-Commerce Sales and Customer Analytics

This script provides a reproducible workflow for:
1. Loading the internship practice dataset
2. Checking data quality
3. Cleaning and standardizing data
4. Handling missing values, duplicates, and negative quantities
5. Calculating KPIs
6. Performing monthly, category, region, product, and customer analysis
7. Performing K-Means customer behavioural clustering
8. Exporting analysis-ready datasets and charts

Run:
    python SahlaFathimaPS_EcommerceSalesAnalytics.py

Internet access is required to load the dataset from the official
spreadsheet export URL.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1GdLk6devNLM_foTrEVNlHFlCC1w4_wqBDKdRG4juDcQ/"
    "export?format=xlsx"
)

OUTPUT_DIR = Path("ecommerce_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def print_section(title):
    """Print a clearly formatted section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def save_csv(dataframe, filename):
    """Save a DataFrame to the output directory."""
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_chart(filename):
    """Save the current matplotlib figure."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------

print_section("E-COMMERCE SALES AND CUSTOMER ANALYTICS USING AI")
print("Loading internship practice dataset...")

try:
    raw = pd.read_excel(DATA_URL)
except Exception as exc:
    raise RuntimeError(
        "The dataset could not be loaded. Check internet access and confirm "
        "that the supplied Google Sheet is available for Excel export."
    ) from exc

print(f"Rows loaded: {len(raw):,}")
print(f"Columns: {list(raw.columns)}")
print("\nFirst five rows:")
print(raw.head())


# ---------------------------------------------------------------------
# 2. INITIAL DATA QUALITY CHECK
# ---------------------------------------------------------------------

print_section("INITIAL DATA QUALITY CHECK")

print("\nMissing values:")
print(raw.isna().sum())

print("\nData types:")
print(raw.dtypes)

if "Order_ID" in raw.columns:
    duplicate_count = int(raw["Order_ID"].duplicated(keep=False).sum())
    print(f"\nRows belonging to duplicate Order_IDs: {duplicate_count:,}")

if "Quantity" in raw.columns:
    quantity_check = pd.to_numeric(raw["Quantity"], errors="coerce")
    negative_count = int((quantity_check < 0).sum())
    print(f"Rows with negative quantity: {negative_count:,}")

quality_report = pd.DataFrame(
    {
        "Column": raw.columns,
        "Missing_Count": [raw[column].isna().sum() for column in raw.columns],
        "Data_Type": [str(raw[column].dtype) for column in raw.columns],
    }
)

save_csv(quality_report, "data_quality_report.csv")


# ---------------------------------------------------------------------
# 3. DATA CLEANING AND PREPARATION
# ---------------------------------------------------------------------

print_section("DATA CLEANING AND PREPARATION")

df = raw.copy()

# Remove leading/trailing spaces from text fields.
for column in ["Order_ID", "Customer_ID", "Product", "Category", "Region"]:
    if column in df.columns:
        df[column] = df[column].astype("string").str.strip()

# Standardize common categorical text fields.
if "Category" in df.columns:
    df["Category"] = df["Category"].str.title()

if "Region" in df.columns:
    df["Region"] = df["Region"].str.title()

# Convert numerical fields safely.
for column in ["Quantity", "Revenue", "Profit"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Validate required columns before continuing.
required_columns = [
    "Order_ID",
    "Customer_ID",
    "Product",
    "Category",
    "Region",
    "Quantity",
    "Revenue",
    "Profit",
    "Order_Date",
]

missing_required = [
    column for column in required_columns if column not in df.columns
]

if missing_required:
    raise ValueError(
        "The dataset is missing required columns: "
        + ", ".join(missing_required)
    )

# Convert dates safely.
df["Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)

cleaning_log = []


def log_cleaning_step(step, before, after):
    """Record the effect of each cleaning operation."""
    cleaning_log.append(
        {
            "Step": step,
            "Rows_Before": before,
            "Rows_After": after,
            "Rows_Removed": before - after,
        }
    )


# Remove records that cannot support the requested analysis.
before = len(df)
df = df.dropna(subset=required_columns[:-1] + ["Date"]).copy()
log_cleaning_step(
    "Remove rows with missing critical fields or invalid dates",
    before,
    len(df),
)

# Negative quantities are treated as invalid sales records for this analysis.
before = len(df)
df = df[df["Quantity"] >= 0].copy()
log_cleaning_step(
    "Remove records with negative quantity",
    before,
    len(df),
)

# Keep one record per Order_ID to avoid duplicate order records.
before = len(df)
df = df.drop_duplicates(subset=["Order_ID"], keep="first").copy()
log_cleaning_step(
    "Remove duplicate Order_ID records",
    before,
    len(df),
)

# Create analysis fields.
df["Month"] = df["Date"].dt.month_name()
df["Month_Number"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

df["Order_Value"] = df["Revenue"]

df["Order_Value_Category"] = pd.cut(
    df["Order_Value"],
    bins=[-np.inf, 1000, 5000, np.inf],
    labels=["Low", "Medium", "High"],
)

# Classify customers according to number of orders.
customer_order_counts = df.groupby("Customer_ID")["Order_ID"].transform("count")

df["Customer_Segment"] = np.where(
    customer_order_counts == 1,
    "New Customer",
    "Repeat Customer",
)

print("\nCleaning summary:")
cleaning_log_df = pd.DataFrame(cleaning_log)
print(cleaning_log_df.to_string(index=False))

save_csv(cleaning_log_df, "cleaning_log.csv")
save_csv(df, "ecommerce_analysis_ready.csv")

print(f"\nFinal analysis-ready rows: {len(df):,}")
print(f"Unique customers: {df['Customer_ID'].nunique():,}")
print(f"Unique orders: {df['Order_ID'].nunique():,}")


# ---------------------------------------------------------------------
# 4. KEY PERFORMANCE INDICATORS
# ---------------------------------------------------------------------

print_section("KEY PERFORMANCE INDICATORS")

total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()

profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue != 0
    else 0
)

unique_customers = df["Customer_ID"].nunique()
unique_orders = df["Order_ID"].nunique()
loss_making_orders = int((df["Profit"] < 0).sum())

kpis = pd.DataFrame(
    {
        "KPI": [
            "Analysis-ready transactions",
            "Unique customers",
            "Unique orders",
            "Total revenue",
            "Total profit",
            "Profit margin (%)",
            "Loss-making orders",
        ],
        "Value": [
            len(df),
            unique_customers,
            unique_orders,
            total_revenue,
            total_profit,
            profit_margin,
            loss_making_orders,
        ],
    }
)

print(kpis.to_string(index=False))
save_csv(kpis, "kpis.csv")


# ---------------------------------------------------------------------
# 5. MONTHLY ANALYSIS
# ---------------------------------------------------------------------

print_section("MONTHLY SALES AND PROFIT ANALYSIS")

monthly = (
    df.groupby(["Month_Number", "Month"], as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "nunique"),
    )
    .sort_values("Month_Number")
)

print(monthly.to_string(index=False))
save_csv(monthly, "monthly_analysis.csv")

plt.figure(figsize=(10, 5))
plt.plot(monthly["Month"], monthly["Revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
save_chart("monthly_revenue_trend.png")


# ---------------------------------------------------------------------
# 6. CATEGORY ANALYSIS
# ---------------------------------------------------------------------

print_section("CATEGORY ANALYSIS")

category = (
    df.groupby("Category", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order_ID", "nunique"),
    )
    .sort_values("Revenue", ascending=False)
)

print(category.to_string(index=False))
save_csv(category, "category_analysis.csv")

plt.figure(figsize=(9, 5))
plt.bar(category["Category"], category["Revenue"])
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
save_chart("revenue_by_category.png")


# ---------------------------------------------------------------------
# 7. REGION ANALYSIS
# ---------------------------------------------------------------------

print_section("REGIONAL ANALYSIS")

region = (
    df.groupby("Region", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order_ID", "nunique"),
    )
    .sort_values("Revenue", ascending=False)
)

print(region.to_string(index=False))
save_csv(region, "region_analysis.csv")

plt.figure(figsize=(10, 5))
plt.bar(region["Region"], region["Revenue"])
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
save_chart("revenue_by_region.png")


# ---------------------------------------------------------------------
# 8. PRODUCT ANALYSIS
# ---------------------------------------------------------------------

print_section("PRODUCT ANALYSIS")

product = (
    df.groupby("Product", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order_ID", "nunique"),
    )
    .sort_values("Revenue", ascending=False)
)

print("\nTop 10 products by revenue:")
print(product.head(10).to_string(index=False))
save_csv(product, "product_analysis.csv")


# ---------------------------------------------------------------------
# 9. CUSTOMER SEGMENTATION
# ---------------------------------------------------------------------

print_section("NEW VS REPEAT CUSTOMER ANALYSIS")

customer_segment = (
    df.groupby("Customer_Segment", as_index=False)
    .agg(
        Orders=("Order_ID", "nunique"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
        Customers=("Customer_ID", "nunique"),
    )
)

print(customer_segment.to_string(index=False))
save_csv(customer_segment, "customer_segment_analysis.csv")


# ---------------------------------------------------------------------
# 10. CUSTOMER-LEVEL SUMMARY
# ---------------------------------------------------------------------

customer_summary = (
    df.groupby("Customer_ID", as_index=False)
    .agg(
        Order_Count=("Order_ID", "nunique"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
    )
)

customer_summary["Average_Order_Value"] = (
    customer_summary["Total_Revenue"]
    / customer_summary["Order_Count"]
)

customer_summary["Customer_Segment"] = np.where(
    customer_summary["Order_Count"] == 1,
    "New Customer",
    "Repeat Customer",
)

customer_summary = customer_summary.sort_values(
    "Total_Revenue",
    ascending=False,
)

save_csv(customer_summary, "customer_summary.csv")


# ---------------------------------------------------------------------
# 11. K-MEANS CUSTOMER BEHAVIOURAL CLUSTERING
# ---------------------------------------------------------------------

print_section("AI/ML: K-MEANS CUSTOMER BEHAVIOURAL CLUSTERING")

features = [
    "Order_Count",
    "Total_Revenue",
    "Total_Profit",
]

if len(customer_summary) >= 3:
    scaler = StandardScaler()
    X = scaler.fit_transform(customer_summary[features])

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    customer_summary["Cluster"] = kmeans.fit_predict(X)

    cluster_summary = (
        customer_summary.groupby("Cluster", as_index=False)
        .agg(
            Customers=("Customer_ID", "count"),
            Avg_Orders=("Order_Count", "mean"),
            Avg_Revenue=("Total_Revenue", "mean"),
            Avg_Profit=("Total_Profit", "mean"),
        )
        .round(2)
    )

    print("\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    save_csv(customer_summary, "customer_ml_segments.csv")
    save_csv(cluster_summary, "customer_cluster_summary.csv")
else:
    print(
        "K-Means skipped because fewer than 3 unique customers "
        "are available."
    )


# ---------------------------------------------------------------------
# 12. BUSINESS INSIGHT SUMMARY
# ---------------------------------------------------------------------

print_section("BUSINESS INSIGHT SUMMARY")

if not monthly.empty:
    best_month_row = monthly.loc[monthly["Revenue"].idxmax()]
    print(
        f"Highest revenue month: {best_month_row['Month']} "
        f"({best_month_row['Revenue']:,.2f})"
    )

if not category.empty:
    best_category = category.iloc[0]
    print(
        f"Highest revenue category: {best_category['Category']} "
        f"({best_category['Revenue']:,.2f})"
    )

if not product.empty:
    best_product = product.iloc[0]
    print(
        f"Highest revenue product: {best_product['Product']} "
        f"({best_product['Revenue']:,.2f})"
    )

if not region.empty:
    best_region = region.iloc[0]
    print(
        f"Highest revenue region: {best_region['Region']} "
        f"({best_region['Revenue']:,.2f})"
    )

print(f"Total revenue: {total_revenue:,.2f}")
print(f"Total profit: {total_profit:,.2f}")
print(f"Profit margin: {profit_margin:.2f}%")
print(f"Loss-making orders: {loss_making_orders:,}")


# ---------------------------------------------------------------------
# 13. COMPLETION MESSAGE
# ---------------------------------------------------------------------

print_section("PROJECT COMPLETE")
print(f"All exported datasets and charts are in: {OUTPUT_DIR.resolve()}")
