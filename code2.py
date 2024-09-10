import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


#Task 1: Data Loading and Exploration

#1. Load the dataset using Pandas.
titanic_df = pd.read_csv('titanic.csv')

#2. Display the first few rows of the dataset to understand its structure.
print("Orignial csv:")
print(titanic_df.head())
print('\n')


#Task 2: Data Preprocessing
#1.Convert categorical variables (Sex and Embarked) into numerical values.

# Convert 'Sex' column to numerical values manually

one_hot = OneHotEncoder()
encoded = one_hot.fit_transform(titanic_df[["Sex"]])
encoded_array = encoded.toarray()
df_encoded = pd.DataFrame(encoded_array, columns= one_hot.get_feature_names_out(["Sex"]))
titanic_df = pd.concat([titanic_df, df_encoded], axis= 1)
titanic_df = titanic_df.drop(columns= ["Sex"])
print("Sex encoded dataframe:")
print(titanic_df.head())

#2.Handle missing values in the Age, Fare, and Embarked columns by filling them with median values.
# Convert 'Embarked' column to dummy variables

# Fill missing values with median values before converting 'Embarked' to dummy variables
if not titanic_df['Age'].dropna().empty:
    age_median = titanic_df['Age'].median()
    titanic_df['Age'] = titanic_df['Age'].fillna(age_median)

if not titanic_df['Fare'].dropna().empty:
    fare_median = titanic_df['Fare'].median()
    titanic_df['Fare'] = titanic_df['Fare'].fillna(fare_median)

if not titanic_df['Embarked'].dropna().empty:
    embarked_mode = titanic_df['Embarked'].mode()[0]
    titanic_df['Embarked'] = titanic_df['Embarked'].fillna(embarked_mode)

# Use on-hot-encoding on non-numerical values
one_hot = OneHotEncoder()
encoded = one_hot.fit_transform(titanic_df[["Embarked"]])
encoded_array = encoded.toarray()
encoded_df = pd.DataFrame(encoded_array, columns= one_hot.get_feature_names_out(["Embarked"]))
titanic_df = pd.concat([titanic_df, encoded_df], axis=1)
titanic_df = titanic_df.drop(columns= ["Embarked"])

#Task 3: Correlation Study
#1. Drop columns that are not useful for correlation: Name, Ticket, Cabin.
titanic_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

print("Numericalized data & useless dropped:")
print(titanic_df.head())
print('\n')

#2. Calculate the correlation matrix for the dataset using the Pandas command: dataframe.corr()

corr_matrix = titanic_df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='plasma', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.savefig("correlation_matrix.png")
plt.show()

titanic_df.to_csv("processed_data.csv")