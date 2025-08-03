import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
train_df = pd.read_csv('titanic/train.csv')

# Preprocessing
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

numerical_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked']
features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = X.drop(features_to_drop, axis=1)

# Model and training
model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestClassifier())])


# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
accuracy = scores.mean()

# Print accuracy
print(f"Accuracy: {accuracy}")