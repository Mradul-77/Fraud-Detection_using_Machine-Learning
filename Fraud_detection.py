#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df=pd.read_excel("Fraud1.xlsx")
df.shape


# In[ ]:


df.head(200)


# In[ ]:


df.tail(200)


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.info()


# In[ ]:


legit = len(df[df.isFraud == 0])
fraud = len(df[df.isFraud == 1])
legit_percent = (legit / (fraud + legit)) * 100
fraud_percent = (fraud / (fraud + legit)) * 100

print("Number of Legit transactions: ", legit)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Legit transactions: {:.4f} %".format(legit_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))


# _These results prove that this is a highly unbalanced data as Percentage of Legit transactions= 99.89 % and Percentage of Fraud transactions= 0.108 %. SO DECISION TREES AND RANDOM FORESTS ARE GOOD METHODS FOR IMBALANCED DATA._

# In[ ]:


X = df[df['nameDest'].str.contains('M')]
X.head()


# _For merchants there is no information regarding the attribites oldbalanceDest and newbalanceDest._

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# ## VISUALISATION

# ##### CORRELATION HEATMAP

# In[ ]:


corr=df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# ##### NUMBER OF LEGIT AND FRAUD TRANSACTIONS

# In[ ]:


plt.figure(figsize=(5,10))
labels = ["Legit", "Fraud"]
count_classes = df.value_counts(df['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# ## PROBLEM SOLVING

# In[ ]:


new_df=df.copy()
new_df.head()


# In[ ]:


objList = new_df.select_dtypes(include = "object").columns
print (objList)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    new_df[feat] = le.fit_transform(new_df[feat].astype(str))

print (new_df.info())


# In[ ]:


new_df.head()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return(vif)

calc_vif(new_df)


# We can see that oldbalanceOrg and newbalanceOrig have too high VIF(VARIANCE INFLATION FACTOR) thus they are highly correlated. Similarly oldbalanceDest and newbalanceDest. Also nameDest is connected to nameOrig.
# 
# Thus combine these pairs of collinear attributes and drop the individual ones.

# In[ ]:


new_df['Actual_amount_orig'] = new_df.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'],axis=1)
new_df['Actual_amount_dest'] = new_df.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'],axis=1)
new_df['TransactionPath'] = new_df.apply(lambda x: x['nameOrig'] + x['nameDest'],axis=1)

#Dropping columns
new_df = new_df.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'],axis=1)

calc_vif(new_df)


# In[ ]:


corr=new_df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True)


# In[ ]:





# ## MODEL BUILDING

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ##### NORMALIZING (SCALING) AMOUNT

# In[ ]:


scaler = StandardScaler()
new_df["NormalizedAmount"] = scaler.fit_transform(new_df["amount"].values.reshape(-1, 1))
new_df.drop(["amount"], inplace= True, axis= 1)

Y = new_df["isFraud"]
X = new_df.drop(["isFraud"], axis= 1)


# I did not normalize the complete dataset because it may lead to decrease in accuracy of model.

# ##### TRAIN-TEST SPLIT

# In[ ]:


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 42)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# ##### MODEL TRAINIG

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test)
decision_tree_score = decision_tree.score(X_test, Y_test) * 100


# In[ ]:


random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)
random_forest_score = random_forest.score(X_test, Y_test) * 100


# ##### EVALUATION

# In[ ]:


print("Decision Tree Score: ", decision_tree_score)
print("Random Forest Score: ", random_forest_score)


# In[ ]:


# key terms of Confusion Matrix - DT

print("TP,FP,TN,FN - Decision Tree")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_dt).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

print("----------------------------------------------------------------------------------------")

# key terms of Confusion Matrix - RF

print("TP,FP,TN,FN - Random Forest")
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_rf).ravel()
print(f'True Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


# TP(Decision Tree) ~ TP(Random Forest) so no competetion here.
# FP(Decision Tree) >> FP(Random Forest) - Random Forest has an edge
# TN(Decision Tree) < TN(Random Forest) - Random Forest is better here too
# FN(Decision Tree) ~ FN(Random Forest)
# 
# Here Random Forest looks good.

# In[ ]:


# confusion matrix - DT

confusion_matrix_dt = confusion_matrix(Y_test, Y_pred_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt,)

print("----------------------------------------------------------------------------------------")

# confusion matrix - RF

confusion_matrix_rf = confusion_matrix(Y_test, Y_pred_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[ ]:


# classification report - DT

classification_report_dt = classification_report(Y_test, Y_pred_dt)
print("Classification Report - Decision Tree")
print(classification_report_dt)

print("----------------------------------------------------------------------------------------")

# classification report - RF

classification_report_rf = classification_report(Y_test, Y_pred_rf)
print("Classification Report - Random Forest")
print(classification_report_rf)


# In[ ]:


# visualising confusion matrix - DT

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dt)
disp.plot()
plt.title('Confusion Matrix - DT')
plt.show()

# visualising confusion matrix - RF
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp.plot()
plt.title('Confusion Matrix - RF')
plt.show()


# In[ ]:


# AUC ROC - DT
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_dt)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - DT')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# AUC ROC - RF
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred_rf)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC - RF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# THE AUC for both Decision Tree and Random Forest is equal, so both models are pretty good at what they do.

# ## CONCLUSION

# We have observed that both the Random Forest and Decision Tree models exhibit the same accuracy, but the Random Forest model demonstrates higher precision. In the context of a fraud detection system, precision is particularly crucial because the priority is to correctly identify fraudulent transactions rather than merely predicting normal transactions accurately. Failing to meet this requirement could result in penalizing innocent individuals and letting actual fraudsters go undetected. This necessity underscores the preference for using Random Forest and Decision Tree models over other algorithms.
# 
# Another reason for selecting these models is the extremely imbalanced nature of the dataset, where legitimate transactions vastly outnumber fraudulent ones (Legit: Fraud :: 99.87:0.13). Random Forest, by creating multiple decision trees, provides a comprehensive approach that facilitates a more nuanced understanding of the data, despite being more time-consuming. This is advantageous over a single Decision Tree, which makes straightforward, binary decisions.

# ### What are the key factors that predict fraudulent customer? 

# Here are the key factors that predict fraudulent customers, presented in a concise manner:
# 
# 1. Unusual Transaction Patterns: Sudden changes in spending behavior, high-value transactions, or frequent transactions in a short time.
# 2. Location-Based Anomalies: Transactions from unexpected or inconsistent geographic locations.
# 3. High Transaction Frequency: Unusually high number of transactions within a short period.
# 4. Atypical Transaction Amounts: Transactions that deviate significantly from usual spending amounts.
# 5. New Account Activity: High transaction volumes in new accounts.
# 6. IP Address Changes: Transactions from IP addresses different from typical ones.
# 7. Device and Browser Changes: Use of new or unusual devices and browsers.
# 8. Frequent Personal Info Changes: Regular updates to personal details like address or phone number.
# 9. Odd Login Times: Logins at unusual hours.
# 10. Failed Login Attempts: Multiple unsuccessful login attempts.
# 11. New Payment Methods: Use of unfamiliar or multiple new payment methods.
# 12. Suspicious Connections: Links to known fraudulent accounts or patterns resembling past fraud cases.
# 
# By tracking these factors, companies can better predict and prevent fraudulent activities.

# ### Do these factors make sense? If yes, How? If not, How not?

# Yes, these factors make sense because they help identify deviations from normal behavior, which are often indicative of fraud:
# 
# 1. Unusual Transaction Patterns: Sudden changes in spending.
# 2. Location-Based Anomalies: Transactions from unexpected locations.
# 3. High Transaction Frequency: Many transactions in a short period.
# 4. Atypical Transaction Amounts: Unusual spending amounts.
# 5. New Account Activity: High activity in new accounts.
# 6. IP Address Changes: Transactions from different IP addresses.
# 7. Device and Browser Changes: Use of new devices or browsers.
# 8. Frequent Personal Info Changes: Regular updates to personal details.
# 9. Odd Login Times: Logins at unusual hours.
# 10. Failed Login Attempts: Multiple unsuccessful logins.
# 11. New Payment Methods: Use of unfamiliar payment methods.
# 12. Suspicious Connections: Links to known fraud patterns.
# 
# Why These Factors Make Sense:
# 
# 1. Anomalies Detection: Identify deviations from normal behavior.
# 2. Pattern Recognition: Use in machine learning models to detect fraud.
# 3. Preventive Measures: Enable timely interventions to prevent fraud.

# ### What kind of prevention should be adopted while company update its infrastructure?

# When updating company infrastructure, the following preventive measures should be adopted:
# 
# 1. Utilize only verified and trusted applications.
# 2. Ensure browsing is conducted through secure websites.
# 3. Use secure internet connections, such as a VPN, to protect data.
# 4. Regularly update security features on both mobile devices and laptops.
# 5. Avoid responding to unsolicited calls, SMS messages, or emails.
# 6. Immediately contact your bank if you suspect any security breach or if you feel you have been deceived.

# ### Assuming these actions have been implemented, how would you determine if they work?

# To determine the effectiveness of these implemented actions, consider the following steps:
# 
# 1. Ensure the bank sends regular e-statements to customers for transparency and monitoring.
# 2. Encourage customers to regularly review their account activity for any unauthorized transactions.
# 3. Maintain a detailed log of all payments to track and verify transactions.
# 4. Monitor for any decrease in security incidents or fraud reports as an indicator of improved security.
# 5. Conduct periodic security audits and assessments to evaluate the robustness of the infrastructure.
# 6. Solicit customer feedback regarding any suspicious activities or security concerns to address potential vulnerabilities promptly.
