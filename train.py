import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the generated dataset
df = pd.read_csv("loan_approval_dataset.csv")

X = df.drop(columns=["approved"])
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define preprocessing pipelines
num_features = ["income", "credit_score"]
num_transformer = StandardScaler()

cat_features = ["employment"]
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ]
)

# 3. Assemble full training workflow
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# 4. Train the model
pipeline.fit(X_train, y_train)

# 5. Export model artifact
joblib.dump(pipeline, "loan_model.joblib")
print("Model artifacts successfully built and exported to loan_model.joblib")

