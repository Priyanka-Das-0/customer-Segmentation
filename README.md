# 🛍️ Customer Segmentation using RFM Analysis and K-Means Clustering

This project aims to segment customers based on purchasing behavior using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering. It helps businesses identify valuable customer groups for targeted marketing.

---

## 📊 Problem Statement

Businesses often struggle to retain customers or target them effectively. This project solves that problem by grouping customers based on:
- **Recency**: How recently a customer made a purchase.
- **Frequency**: How often they purchase.
- **Monetary**: How much money they spend.

---

## 📁 Dataset

- Source: [Online Retail Data Set](https://www.kaggle.com/datasets/ulrikthygepedersen/online-retail-dataset)
- Contains ~540K transaction records from a UK-based online store between 2010 and 2011.

---

## 🧰 Technologies Used

- Python (Pandas, NumPy, Seaborn, Matplotlib, Plotly, scikit-learn)
- Jupyter Notebook / VS Code
- Streamlit (Optional frontend)
- KMeans Clustering
- RFM Scoring Model

---

## 🔍 Steps Performed

1. **Data Cleaning**
   - Removed null `CustomerID` rows
   - Filtered for UK transactions only
   - Removed negative `Quantity` and `UnitPrice` values

2. **Feature Engineering**
   - Calculated `TotalAmount = Quantity × UnitPrice`
   - Performed RFM scoring
   - Labeled customers as Platinum, Gold, Silver, Bronze

3. **Clustering**
   - Scaled and log-transformed data
   - Used Elbow Method to determine optimal number of clusters (k)
   - Applied K-Means to group similar customers
   - Visualized clusters in 2D and 3D

4. **Visualization**
   - 2D scatter plots: Recency vs Frequency, Frequency vs Monetary
   - 3D plot: RFM clustering using Plotly

---

## 📌 Key Insights

- Customers with recent and frequent purchases and high spend were segmented into high-value groups (Platinum).
- Elbow method indicated optimal clusters (e.g., `k=3`).
- Helps in building targeted marketing strategies and improving customer retention.

---


