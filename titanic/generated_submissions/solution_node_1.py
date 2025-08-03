import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
try:
    train_df = pd.read_csv('titanic/train.csv')
except FileNotFoundError:
    print("Error: 'titanic/train.csv' not found. Please ensure the file is in the correct location.")
    exit()


# 2. Preprocess & Feature Engineer
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# 3. Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LogisticRegression(max_iter=1000)  # Increased max_iter
model.fit(X_train, y_train)


# 5. Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# 6. Output
print(f"Accuracy: {accuracy}")