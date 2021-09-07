import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import k_means
from sklearn.decomposition import PCA

### This is data from a bank about it's customers over the course of 6 months
### Goal: Use ML in order to launch a targeted marketing ad campaign tailered to specific segments
### Want to divide customers into new customers, customers who use credit cards for transactions only, customers who use their cards for loans, and customers who are increasing their credit limit

# Variable explanations:
### CUSTID: Identification of Credit Card holder 
### BALANCE: Balance amount left in customer's account to make purchases
### BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
### PURCHASES: Amount of purchases made from account
### ONEOFFPURCHASES: Maximum purchase amount done in one-go
### INSTALLMENTS_PURCHASES: Amount of purchase done in installment
### CASH_ADVANCE: Cash in advance given by the user
### PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
### ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
### PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
### CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
### CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
### PURCHASES_TRX: Number of purchase transactions made
### CREDIT_LIMIT: Limit of Credit Card for user
### PAYMENTS: Amount of Payment done by user
### MINIMUM_PAYMENTS: Minimum amount of payments made by user  
### PRC_FULL_PAYMENT: Percent of full payment paid by user
### TENURE: Tenure of credit card service for user

df = pd.read_csv('marketing_data.csv')
df.head()
df.info()
df.describe()

### See how many missing valus there were in the data and impute them with average
sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap = 'Reds')
df.isnull().sum()
df.loc[(df.MINIMUM_PAYMENTS.isnull() == True), 'MINIMUM_PAYMENTS'] = df.MINIMUM_PAYMENTS.mean()
df.loc[(df.CREDIT_LIMIT.isnull() == True), 'CREDIT_LIMIT'] = df.MINIMUM_PAYMENTS.mean()

df.duplicated().sum()

### Remove irrelevant variable
df.drop('CUST_ID',axis=1, inplace = True)

### KDE demonstrates the probability density at different values in a continuous variable. 
plt.figure(figsize=(10,50))
for i in range(len(df.columns)):
  plt.subplot(17,1,i+1)
  sns.distplot(df[df.columns[i]], kde_kws={'color':'b', 'lw':3, 'label':'KDE'}, hist_kws={'color':'g'})
  plt.title(df.columns[i])

plt.tight_layout()
