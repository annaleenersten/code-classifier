import pandas as pd
import numpy as np
import gc
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load data cleanly (try 'vuln_train.csv' / 'vuln_test.csv' or 'bugs_train.csv' / 'bugs_test.csv')
train_data = pd.read_csv('data/processed/vuln_train.csv', usecols=['code', 'label']).dropna()
test_data = pd.read_csv('data/processed/vuln_test.csv', usecols=['code', 'label']).dropna()

# 2. Extract smarter structural & operator features
def extract_code_metrics(df):
    code = df['code'].astype(str)
    metrics = np.column_stack([
        code.str.len().values,                         # Character count
        code.str.count('\n').values + 1,               # Line count
        code.str.count('if ').values,                  # Branching
        code.str.count('==').values,                   # Equality checks
        code.str.count('!=').values,                   # Inequality checks
        code.str.count('None').values,                 # Null checks
        code.str.count(r'\[').values,                  # List/dict indexing (IndexError/KeyError risk)
        code.str.count('    ').values                  # Indentation/nesting complexity
    ]).astype(np.float32)
    return csr_matrix(metrics)  # Convert to sparse format to save RAM

X_train_metrics = extract_code_metrics(train_data)
X_test_metrics = extract_code_metrics(test_data)

# 3. Extract the top 500 code tokens using TF-IDF (captures actual keywords & functions)
vectorizer = TfidfVectorizer(max_features=500, token_pattern=r'\b\w+\b')
X_train_tokens = vectorizer.fit_transform(train_data['code'].astype(str))
X_test_tokens = vectorizer.transform(test_data['code'].astype(str))

# 4. Combine structural metrics + code tokens into our upgraded X matrix
X_train = hstack([X_train_metrics, X_train_tokens])
y_train = train_data['label'].values.astype(np.int8)

X_test = hstack([X_test_metrics, X_test_tokens])
y_test = test_data['label'].values.astype(np.int8)

# Clear raw text tables from RAM
del train_data, test_data
gc.collect()

# 5. Train the Random Forest
rf_engine = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
rf_engine.fit(X_train, y_train)

# 6. Evaluate Results
preds = rf_engine.predict(X_test)
probs = rf_engine.predict_proba(X_test)[:, 1]

print(f"Upgraded Test Accuracy: {accuracy_score(y_test, preds):.2%}")
print(f"Upgraded ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

