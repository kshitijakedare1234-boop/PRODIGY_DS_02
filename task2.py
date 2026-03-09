import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%202/train.csv"
data = pd.read_csv(url)
print("Dataset shape:", data.shape)
print("Columns:", data.columns)
print(data.head())
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data.drop(['Cabin'], axis=1, inplace=True)

sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()


sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("Survival by Gender")
plt.show()


sns.histplot(data['Age'], bins=20, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()


sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("Survival by Passenger Class")
plt.show()


sns.countplot(x='Embarked', hue='Survived', data=data)
plt.title("Survival by Embarkation Point")
plt.show()


