### This is data from a bank about it's customers over the course of 6 months
https://www.kaggle.com/arjunbhasin2013/ccdata
### Goal: Use ML in order to launch a targeted marketing ad campaign tailered to specific segments
### Want to divide customers into new customers, customers who use credit cards for transactions only, customers who use their cards for loans, and customers who are increasing their credit limit

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
```

### Variable explanations:
```
 CUSTID: Identification of Credit Card holder 
 BALANCE: Balance amount left in customer's account to make purchases
 BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
 PURCHASES: Amount of purchases made from account
 ONEOFFPURCHASES: Maximum purchase amount done in one-go
 INSTALLMENTS_PURCHASES: Amount of purchase done in installment
 CASH_ADVANCE: Cash in advance given by the user
 PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
 ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
 PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
 CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
 CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
 PURCHASES_TRX: Number of purchase transactions made
 CREDIT_LIMIT: Limit of Credit Card for user
 PAYMENTS: Amount of Payment done by user
 MINIMUM_PAYMENTS: Minimum amount of payments made by user  
 PRC_FULL_PAYMENT: Percent of full payment paid by user
 TENURE: Tenure of credit card service for user
```
```python
df = pd.read_csv('marketing_data.csv')
df.head()
df.info()
df.describe()
```
### See how many missing valus there were in the data and impute them with average
```python
sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap = 'Reds')
df.isnull().sum()
df.loc[(df.MINIMUM_PAYMENTS.isnull() == True), 'MINIMUM_PAYMENTS'] = df.MINIMUM_PAYMENTS.mean()
df.loc[(df.CREDIT_LIMIT.isnull() == True), 'CREDIT_LIMIT'] = df.MINIMUM_PAYMENTS.mean()

df.duplicated().sum()
```
![image](https://user-images.githubusercontent.com/86034623/132415948-28c9be85-4e4a-4eec-913f-d16d49928f23.png)

### Remove irrelevant variable
```python
df.drop('CUST_ID',axis=1, inplace = True)
```
### KDE demonstrates the probability density at different values in a continuous variable.
```python
plt.figure(figsize=(10,50))
for i in range(len(df.columns)):
  plt.subplot(17,1,i+1)
  sns.distplot(df[df.columns[i]], kde_kws={'color':'b', 'lw':3, 'label':'KDE','bw':0.1}, hist_kws={'color':'g'})
  plt.title(df.columns[i])

plt.tight_layout()
```
![image](https://user-images.githubusercontent.com/86034623/132416048-c4bee91a-ace6-48c7-ba34-694160cbdbd6.png)

### This gives us deeper look into how all the variables are connected and if we were running a ML model we would probably remove highly correlated (>0.7) to prevent multicollinearity
```python
plt.subplots(figsize = (20,10))
sns.heatmap(df.corr(), annot=True, mask=np.triu(df.corr()))
```
![image](https://user-images.githubusercontent.com/86034623/132416110-bfb3a813-287f-4e34-b9de-7d5c211aed88.png)

### Here we are standardizing the data so that they all are between 0 and 1
```python
scaler = StandardScaler()
df1 = scaler.fit_transform(df)
```
### Looking at the chart below, we can see that the elbow lands at about 4 clusters
```python
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df1)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(12,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```
![image](https://user-images.githubusercontent.com/86034623/132415789-298341dd-b7c1-4bee-b880-bbaed67cbb8d.png)

### When we try to look again at the data, we'll notice that it is difficult to read when scaled.
```python
kmeans = KMeans(4)
kmeans.fit(df1)
labels = kmeans.labels_

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df.columns])
cluster_centers
```
### Here we are inversing/removing the standardization since we found our elbow
```python
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df.columns])
cluster_centers
```
```
First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 
```
```python
# concatenate the clusters labels to our original dataframe so that each row has an associated cluster they fit into
df_cluster = pd.concat([df, pd.DataFrame({'cluster':labels})], axis = 1)
df_cluster.head()
```


### Plot the histogram of various clusters, helps confirm different clusters created
```python
# Normally this would show the four clusters for every variables, but here we only look at one
for i in df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(4):
    plt.subplot(1,4,j+1)
    cluster = df_cluster[df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()
```
![image](https://user-images.githubusercontent.com/86034623/132416553-61a81d28-bfd6-4089-91a5-c02151b5ee09.png)

```python
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(df1)
principal_comp

# Create two PCAs
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()

# Concat thes into data frame
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()
```

### With this we can view the the clusters all together and see where they differ and take place
```python
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','purple'])
plt.show()
```
![image](https://user-images.githubusercontent.com/86034623/132416977-965e1591-2e5c-4996-82fd-6b279ac5730e.png)

### Recap
```
# Perfromed data vizualizations, fixed missing values
# Applied kmeans to better understand customer segmentation
# Plotted histogram distribution of all various clusters
# Used PCA to convert from out original data into a component space and be able to visualized the different clusters in that way
```
