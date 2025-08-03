import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 1. Load Data
train_df = pd.read_csv('titanic/train.csv')

# 2. Preprocess & Feature Engineer
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 8, "Mlle": 9, "Mme": 10,
                "Don": 11, "Dona": 12, "Lady": 13, "Countess": 14, "Jonkheer": 15, "Sir": 16, "Capt": 17, "Ms": 18, "the Countess": 19, "Lady" : 20}
train_df['Title'] = train_df['Title'].map(title_mapping)
train_df['Title'] = train_df['Title'].fillna(0)

train_df['Age'] = train_df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

train_df['Embarked'] = train_df['Embarked'].fillna('S')
embarked_ohe = pd.get_dummies(train_df['Embarked'], prefix='Embarked')
train_df = pd.concat([train_df, embarked_ohe], axis=1)
train_df.drop('Embarked', axis=1, inplace=True)

train_df['Cabin'] = train_df['Cabin'].fillna('U')
train_df['Deck'] = train_df['Cabin'].str.get(0)
deck_mapping = {"U": 0, "C": 1, "B": 2, "D": 3, "E": 4, "A": 5, "F": 6, "G": 7, "T": 8}
train_df['Deck'] = train_df['Deck'].map(deck_mapping)

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['FarePerPerson'] = train_df['Fare'] / train_df['FamilySize']
ticket_counts = train_df['Ticket'].value_counts()
train_df['TicketFrequency'] = train_df['Ticket'].map(ticket_counts)



features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Deck', 'FarePerPerson', 'TicketFrequency']
target = 'Survived'
X = train_df[features]
y = train_df[target]


# 3. Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize numerical features
numerical_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 'TicketFrequency']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])

# 4. Train Model & Hyperparameter Tuning
gb_model = GradientBoostingClassifier(random_state=42)
param_dist = {
    'n_estimators': stats.randint(50, 500),
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, None],
    'min_samples_split': stats.randint(2, 21),
    'min_samples_leaf': stats.randint(1, 11),
    'max_features': ['sqrt', 'log2', None],
}


random_search = RandomizedSearchCV(gb_model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
best_gb_model = random_search.best_estimator_




# 5. Evaluate
y_pred = best_gb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# 6. Output
print(f"Accuracy: {accuracy}")