# utils/ml_models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report

def train_ml_models_regression(df):
    """
    Regression models to predict 'attractive' strikes (explainable).
    Returns models' results dict and recommended top calls/puts by prediction.
    """
    if df.empty or len(df) < 10:
        return {}, [], [], {}
    
    # Create target variable based on OI and price action
    df_ml = df.copy()
    df_ml['TARGET'] = df_ml['CALL_OI'] * df_ml['CALL_LTP'] - df_ml['PUT_OI'] * df_ml['PUT_LTP']
    
    X = df_ml[['CALL_OI', 'PUT_OI', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI', 'CALL_IV', 'PUT_IV', 'CALL_LTP', 'PUT_LTP']].fillna(0)
    y = df_ml['TARGET']
    
    # Feature importance using Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=150, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"model": model, "mae": mae, "r2": r2, "scaler": scaler}

    best_name = min(results, key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']
    best_scaler = results[best_name]['scaler']

    X_full = scaler.transform(X)
    df_ml['ML_PREDICTED_VALUE'] = best_model.predict(X_full)
    
    # Get top calls and puts based on ML predictions
    top_calls = df_ml.nlargest(3, 'ML_PREDICTED_VALUE')['STRIKE'].tolist()
    top_puts = df_ml.nsmallest(3, 'ML_PREDICTED_VALUE')['STRIKE'].tolist()

    return results, top_calls, top_puts, feature_importance

def train_ml_models_classification(df):
    """
    Classification to detect bias from options features.
    Simple labeled by PCR thresholds.
    """
    if df.empty or len(df) < 12:
        return {"RF": 0.0, "LR": 0.0}, "Neutral", {}
    
    d = df.copy()
    d['PCR'] = d['PUT_OI'] / (d['CALL_OI'] + 1)
    d['IVS'] = d['CALL_IV'] - d['PUT_IV']
    d['FLOW_RATIO'] = (d['CALL_CHNG_IN_OI'] * d['CALL_LTP']) / (d['PUT_CHNG_IN_OI'] * d['PUT_LTP'] + 1)
    
    features = ['CALL_OI', 'PUT_OI', 'PCR', 'IVS', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI', 'FLOW_RATIO']
    d['LABEL'] = np.where(d['PCR'] > 1.2, 'Bullish', np.where(d['PCR'] < 0.8, 'Bearish', 'Neutral'))
    
    # Handle the ValueError by ensuring all classes have at least 2 members
    y_counts = d['LABEL'].value_counts()
    valid_classes = y_counts[y_counts >= 2].index
    d_filtered = d[d['LABEL'].isin(valid_classes)]
    
    if len(d_filtered) < 12 or len(d_filtered['LABEL'].unique()) < 2:
        return {"RF": 0.0, "LR": 0.0}, "Neutral", {}

    X = d_filtered[features].fillna(0)
    y = d_filtered['LABEL']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    feature_importance = dict(zip(features, rf.feature_importances_))
    
    lr = LogisticRegression(max_iter=500, class_weight='balanced')
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    
    # consensus label from RF predictions on the set (mode)
    preds = rf.predict(X)
    consensus = pd.Series(preds).mode().iloc[0] if len(preds) > 0 else "Neutral"
    
    # Confusion matrix and classification report
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=['Bullish', 'Neutral', 'Bearish'])
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {"RF": round(acc_rf, 3), "LR": round(acc_lr, 3)}, consensus, {
        "feature_importance": feature_importance,
        "confusion_matrix": cm,
        "classification_report": cr
    }