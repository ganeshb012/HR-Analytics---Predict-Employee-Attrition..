import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\badgu\OneDrive\Desktop\Tasks Datasets\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 3. Data Cleaning
df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

# 4. EDA (Exploratory Data Analysis)

# Attrition Count
sns.countplot(x='Attrition', data=df)
plt.title('Overall Attrition Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.savefig("attrition_count.png")
plt.show()

# Overtime vs Attrition
sns.countplot(x='OverTime_Yes', hue='Attrition', data=df)
plt.title('Attrition by Overtime')
plt.xticks([0, 1], ['No Overtime', 'Overtime'])
plt.savefig("overtime_vs_attrition.png")
plt.show()

# Monthly Income Boxplot
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Monthly Income vs Attrition')
plt.xticks([0, 1], ['No', 'Yes'])
plt.savefig("income_vs_attrition.png")
plt.show()

# Age Histogram
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, element='step')
plt.title('Age Distribution by Attrition')
plt.savefig("age_distribution.png")
plt.show()

# 5. Model Building
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 6. SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 7. Export Cleaned Data for Tableau
df.to_csv("cleaned_hr_data.csv", index=False)
print("✅ Cleaned data saved for Tableau: cleaned_hr_data.csv")
