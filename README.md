# E-Commerce Sales and Customer Analytics Using AI

## IBM SkillsBuild Data Analytics with AI Academic Internship

**Author:** Sahla Fathima P S  
**Project:** E-Commerce Sales and Customer Analytics

## Project Overview

This project performs an end-to-end analysis of an e-commerce transaction dataset supplied with the IBM SkillsBuild Data Analytics with AI internship learning material.

The workflow covers:
- Data-quality assessment
- Data cleaning and preparation
- Sales and profitability KPI calculation
- Monthly, category, product and regional analysis
- New vs. repeat customer analysis
- Customer-level analysis
- K-Means customer behavioural clustering
- Export of analysis-ready and summary datasets

## Dataset

IBM SkillsBuild Masterclass 1 practice dataset:
https://docs.google.com/spreadsheets/d/1GdLk6devNLM_foTrEVNlHFlCC1w4_wqBDKdRG4juDcQ/export?format=xlsx

Main fields: `Order_ID`, `Order_Date`, `Customer_ID`, `Product`, `Category`, `Region`, `Quantity`, `Revenue`, `Profit`.

## Data Preparation

The workflow checks missing values and types, standardizes text, converts numeric fields and dates safely, excludes records that cannot support the main analysis, removes duplicate `Order_ID` records, removes negative-quantity records from sales analysis, retains negative profit values as potentially valid loss-making transactions, and creates derived fields such as month, year, order value and customer segment.

The code does not arbitrarily truncate the dataset to a fixed row count.

## Analysis and AI/ML

The project calculates revenue, profit, profit margin, customers, orders and loss-making orders; analyses monthly/category/product/region performance; classifies customers as New Customer or Repeat Customer; and applies K-Means to standardized `Order_Count`, `Total_Revenue` and `Total_Profit` using three clusters.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook and openpyxl.

## Project Files

- `SahlaFathimaPS_EcommerceSalesAnalytics.ipynb` — complete notebook/code
- `SahlaFathimaPS_EcommerceSalesAnalytics.py` — Python script version
- `requirements.txt` — required Python libraries
- `README.md` — project overview and setup instructions
- `SahlaFathimaPS_ProjectReport.docx` — project report

## Setup

```bash
pip install -r requirements.txt
python SahlaFathimaPS_EcommerceSalesAnalytics.py
```

Or open the notebook in Jupyter. Internet access is required because the code loads the spreadsheet directly from the dataset export URL.

## Output

Running the project creates an `ecommerce_outputs` folder containing the cleaning log, analysis-ready data, KPI summary, analytical summary CSV files, charts and customer cluster results.

## Consistency Note

The notebook and Python script use the same dataset URL and analysis workflow. The project report documents the reference analysis results; the code recomputes the results from the source dataset rather than hard-coding the reported numbers.

## Limitations

The project uses a practice dataset rather than a live company database. Results are specific to this dataset and should not be interpreted as real-world company performance.
