import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

cvs_df = pd.read_csv("synthetic_health_data.csv")

# Initial Data Inspection
cvs_df.head()
cvs_df.shape
cvs_df.info()
cvs_df.describe()
cvs_df.isnull().sum()
cvs_df.dropna(inplace=True)

plt.figure(figsize=(20,8))
cvs_df.boxplot()

# Checking unique values in categorical columns
non_numeric_columns = cvs_df.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    print(f"Unique values in {column}: {cvs_df[column].unique()}")

# Correlation Heatmap
numerical_df = cvs_df.select_dtypes(include=[np.number])
plt.figure(figsize=(14,14))
correlation = numerical_df.corr()
sns.heatmap(abs(correlation), annot=True)

# Visualizations for Risk Analysis
my_palette = {0:'orange', 1:'teal'}

plt.figure(figsize=(10,10))
sns.countplot(x=cvs_df['sex'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Which gender is more prone to CHD?")
plt.legend(['No Risk', 'At Risk'])
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x=cvs_df['diabetes'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Are diabetic patients at more risk of coronary heart disease?")
plt.legend(['No Risk', 'At Risk'])
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x=cvs_df['is_smoking'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Are smokers at more risk of CHD?")
plt.legend(['No Risk', 'At Risk'])
plt.show()

plt.figure(figsize=(10,10))
sns.countplot(x=cvs_df['prevalentHyp'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Are hypertensive patients at more risk of CHD?")
plt.legend(['No Risk', 'At Risk'])
plt.show()

plt.figure(figsize=(10,10))
ax = sns.boxplot(x=cvs_df['sex'], y=cvs_df['age'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Which Age Group is more vulnerable to CHD?")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['No Risk', 'At Risk'], loc="best")
plt.show()

# Heart Rate Labeling
def heartRate_data(row):
    if row['heartRate'] <= 59:
        return 'Low'
    elif row['heartRate'] < 100:
        return 'Normal'
    else:
        return 'High'

cvs_df['heartRateLabel'] = cvs_df.apply(heartRate_data, axis=1)

plt.figure(figsize=(10,10))
sns.countplot(x=cvs_df['heartRateLabel'], hue=cvs_df['TenYearCHD'], palette=my_palette)
plt.title("Is heart rate responsible for CHD?")
plt.legend(['No Risk', 'At Risk'])
plt.show()

cvs_df.drop(columns=['heartRateLabel'], inplace=True)

# Blood Pressure Classification
def blood_pressure_classification(SysBP, DiaBP):
    if SysBP < 90 or DiaBP < 60:
        return 0
    elif SysBP < 120 and DiaBP < 80:
        return 1
    elif SysBP < 130 and DiaBP < 85:
        return 2
    elif SysBP < 140 and DiaBP < 90:
        return 3
    elif SysBP < 160 and DiaBP < 100:
        return 4
    elif SysBP < 180 and DiaBP < 110:
        return 5
    else:
        return 6

cvs_df['Hypertension'] = cvs_df.apply(lambda x: blood_pressure_classification(x['sysBP'], x['diaBP']), axis=1)

# Diabetes Classification
def diabetes_grade(glucose):
    if glucose < 100:
        return 1
    elif glucose < 125:
        return 2
    elif glucose < 200:
        return 3
    else:
        return 4

cvs_df['Diabetes'] = cvs_df['glucose'].apply(diabetes_grade)

# Log transformation for smoking factor
cvs_df['SmokingFactor'] = cvs_df['cigsPerDay'].apply(lambda x: 0 if x < 1 else np.log(x))

# Dropping redundant columns
cvs_df.drop(columns=['prevalentHyp', 'sysBP', 'diaBP', 'glucose', 'diabetes', 'is_smoking', 'cigsPerDay', 'BPMeds', 'prevalentStroke'], inplace=True)

# Encoding categorical variable 'sex' if it's not numeric
if cvs_df['sex'].dtype == 'object':
    cvs_df['sex'] = cvs_df['sex'].apply(lambda x: 1 if x == 'M' else 0)

# Splitting data into features and target variable
dependent_variable = 'TenYearCHD'
independent_variable = list(cvs_df.columns)
independent_variable.remove(dependent_variable)

X = cvs_df[independent_variable].values
y = cvs_df[dependent_variable].values

# Checking class distribution
counter = Counter(y)
print('Before SMOTE:', counter)

# Splitting data before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE only on training data
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After SMOTE:', counter)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# Model Training
decision_tree_model = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=50, random_state=0)
decision_tree_model.fit(X_train_scaled, y_train_sm)

# Predictions
train_preds = decision_tree_model.predict(X_train_scaled)
test_preds = decision_tree_model.predict(X_test_scaled)

# Accuracy
training_accuracy = accuracy_score(y_train_sm, train_preds) * 100
print(f'The training accuracy is {training_accuracy:.2f}%')

testing_accuracy = accuracy_score(y_test, test_preds) * 100
print(f'The testing accuracy is {testing_accuracy:.2f}%')

# Confusion Matrix - Training Data
labels = ['No Risk', 'At Risk']
cm = confusion_matrix(y_train_sm, train_preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for Training Data')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()

# Confusion Matrix - Testing Data
cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for Testing Data')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()

# Classification Reports
print("Classification Report (Training Data):")
print(classification_report(y_train_sm, train_preds))

print("Classification Report (Testing Data):")
print(classification_report(y_test, test_preds))
