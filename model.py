# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as gobj
import plotly.offline as po
import plotly.express as px

Rtl_data = pd.read_csv('online_retail.csv', encoding='unicode_escape')
print("Dataset loaded. Shape:", Rtl_data.shape)

print("Customer distribution by country:")
country_cust_data = Rtl_data[['Country', 'CustomerID']].drop_duplicates()
print(country_cust_data.groupby(['Country'])['CustomerID'].count().sort_values(ascending=False))

# Keep only United Kingdom
Rtl_data = Rtl_data.query("Country=='United Kingdom'").reset_index(drop=True)
print("Filtered UK data. Shape:", Rtl_data.shape)

#
Rtl_data = Rtl_data[pd.notnull(Rtl_data['CustomerID'])]
print("Removed missing CustomerID. Shape:", Rtl_data.shape)

Rtl_data = Rtl_data[Rtl_data['Quantity'] > 0]
Rtl_data = Rtl_data[Rtl_data['UnitPrice'] >= 0]
print("Filtered invalid Quantity/UnitPrice. Shape:", Rtl_data.shape)

Rtl_data['InvoiceDate'] = pd.to_datetime(Rtl_data['InvoiceDate'])
Rtl_data['TotalAmount'] = Rtl_data['Quantity'] * Rtl_data['UnitPrice']

Latest_Date = dt.datetime(2011, 12, 10)
RFMScores = Rtl_data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (Latest_Date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
})
RFMScores.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'}, inplace=True)
print("Created RFM table. Shape:", RFMScores.shape)

quantiles = RFMScores.quantile(q=[0.25, 0.5, 0.75]).to_dict()

def RScoring(x, p, d):
    if x <= d[p][0.25]: return 1
    elif x <= d[p][0.50]: return 2
    elif x <= d[p][0.75]: return 3
    else: return 4

def FnMScoring(x, p, d):
    if x <= d[p][0.25]: return 4
    elif x <= d[p][0.50]: return 3
    elif x <= d[p][0.75]: return 2
    else: return 1

RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency', quantiles))
RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency', quantiles))
RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary', quantiles))


RFMScores['RFMGroup'] = RFMScores['R'].astype(str) + RFMScores['F'].astype(str) + RFMScores['M'].astype(str)
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis=1)

Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
RFMScores['RFM_Loyalty_Level'] = pd.qcut(RFMScores['RFMScore'], q=4, labels=Loyalty_Level)
print("RFM Scoring complete.")

print("Plotting Recency vs Frequency...")
graph = RFMScores[RFMScores['Monetary'] < 50000]
trace_data = []

for level, color, size in zip(Loyalty_Level, ['black', 'red', 'green', 'blue'], [13, 11, 9, 7]):
    trace_data.append(
        gobj.Scatter(
            x=graph[graph['RFM_Loyalty_Level'] == level]['Recency'],
            y=graph[graph['RFM_Loyalty_Level'] == level]['Frequency'],
            mode='markers', name=level,
            marker=dict(size=size, color=color, opacity=0.7, line=dict(width=1))
        )
    )

layout = gobj.Layout(title='Recency vs Frequency', xaxis=dict(title='Recency'), yaxis=dict(title='Frequency'))
po.plot(gobj.Figure(data=trace_data, layout=layout), filename='recency_vs_frequency.html')

def handle_neg_n_zero(num): return 1 if num <= 0 else num

RFMScores['Recency'] = RFMScores['Recency'].apply(handle_neg_n_zero)
RFMScores['Monetary'] = RFMScores['Monetary'].apply(handle_neg_n_zero)

print("Log transforming and scaling data...")
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log).round(3)
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)
Scaled_Data = pd.DataFrame(Scaled_Data, index=RFMScores.index, columns=Log_Tfd_Data.columns)

print("Finding optimal k using Elbow method...")
sum_of_sq_dist = {}
for k in range(1, 15):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, random_state=42)
    km.fit(Scaled_Data)
    sum_of_sq_dist[k] = km.inertia_

plt.figure(figsize=(8,5))
sns.pointplot(x=list(sum_of_sq_dist.keys()), y=list(sum_of_sq_dist.values()))
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_plot.png')
plt.close()

print("Running final KMeans clustering...")
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, random_state=42)
RFMScores['Cluster'] = kmeans.fit_predict(Scaled_Data)

Colors = ["red", "green", "blue"]
RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])

print("Saving cluster scatter plot (Recency vs Frequency)...")
plt.figure(figsize=(10, 8))
plt.scatter(RFMScores['Recency'], RFMScores['Frequency'], c=RFMScores['Color'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('KMeans Clustering - Recency vs Frequency')
plt.savefig('kmeans_recency_frequency.png')
plt.close()

RFMScores.to_csv('Final_RFM_Segmentation.csv')
print("Segmentation complete. Results saved to Final_RFM_Segmentation.csv")
Log_Tfd_Data['Cluster'] = RFMScores['Cluster']

fig = px.scatter_3d(
    Log_Tfd_Data,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Cluster',
    opacity=0.7,
    title='3D Cluster Plot (Log-Transformed RFM)',
)

fig.show()
