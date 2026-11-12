"""
Sentinel URL - Phase 2, Step 1
ML Model Comparison: Random Forest vs XGBoost vs Logistic Regression
Fixed for 4-class (multiclass) classification
Run: python step1_model_comparison.py
"""

import pandas as pd
import numpy as np
import re
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(url):
    url = str(url)
    features = {}
    features["url_length"]        = len(url)
    features["dot_count"]         = url.count(".")
    features["hyphen_count"]      = url.count("-")
    features["slash_count"]       = url.count("/")
    features["at_count"]          = url.count("@")
    features["question_count"]    = url.count("?")
    features["equal_count"]       = url.count("=")
    features["underscore_count"]  = url.count("_")
    features["percent_count"]     = url.count("%")
    features["digit_count"]       = sum(c.isdigit() for c in url)
    features["has_ip"]            = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    features["has_https"]         = 1 if url.startswith("https") else 0
    features["has_http"]          = 1 if url.startswith("http://") else 0
    features["has_double_slash"]  = 1 if "//" in url[7:] else 0
    features["has_at_symbol"]     = 1 if "@" in url else 0
    try:
        domain = url.split("/")[2] if "/" in url else url
        prob   = [domain.count(c) / len(domain) for c in set(domain)]
        entropy = -sum(p * np.log2(p) for p in prob if p > 0)
    except:
        entropy = 0
    features["domain_entropy"]    = round(entropy, 4)
    features["url_depth"]         = url.count("/") - 2 if url.count("/") > 2 else 0
    suspicious_words = ["login","signin","verify","secure","update",
                        "bank","account","confirm","password","free"]
    features["suspicious_keyword"] = 1 if any(w in url.lower() for w in suspicious_words) else 0
    suspicious_tlds = [".xyz",".tk",".ml",".ga",".cf",".gq",".top",".vip"]
    features["suspicious_tld"]    = 1 if any(url.lower().endswith(t) for t in suspicious_tlds) else 0
    return features

# ─────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset(path="encoded_dataset.csv"):
    print(f"\n[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    url_col   = next((c for c in df.columns if "url"   in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "label" in c.lower() or
                      "class" in c.lower() or "target" in c.lower()), df.columns[1])
    print(f"[INFO] Using columns: URL='{url_col}', Label='{label_col}'")
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Class distribution:\n{df[label_col].value_counts()}")
    return df[url_col].astype(str).tolist(), df[label_col].tolist()

# ─────────────────────────────────────────────
# 3. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────

def build_feature_matrix(urls):
    print("\n[INFO] Extracting features from URLs...")
    print("[INFO] Please wait — this may take 3-5 minutes for 640k URLs...")
    features_list = []
    total = len(urls)
    for i, url in enumerate(urls):
        features_list.append(extract_features(url))
        if (i+1) % 100000 == 0:
            print(f"[INFO] Progress: {i+1:,} / {total:,} ({(i+1)/total*100:.1f}%)")
    df = pd.DataFrame(features_list)
    print(f"[INFO] Feature matrix shape: {df.shape}")
    return df

# ─────────────────────────────────────────────
# 4. TRAIN & COMPARE MODELS
# ─────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (GBM)":       GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results    = {}
    best_model = None
    best_f1    = 0

    print("\n" + "="*60)
    print("  MODEL COMPARISON RESULTS")
    print("="*60)

    for name, model in models.items():
        print(f"\n  Training {name} ... (please wait)")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)

        # ✅ FIXED: average='weighted' for multiclass
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        # ✅ FIXED: multi_class='ovr' for multiclass ROC-AUC
        roc_auc   = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='weighted')

        results[name] = {
            "model":      model,
            "accuracy":   acc,
            "precision":  precision,
            "recall":     recall,
            "f1_score":   f1,
            "roc_auc":    roc_auc,
            "train_time": elapsed,
        }

        print(f"\n  {name}")
        print(f"  {'─'*40}")
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"  Train time: {elapsed:.2f}s")

        if f1 > best_f1:
            best_f1    = f1
            best_model = (name, model)

    return results, best_model

# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────

def print_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n  Full Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Benign(0)", "Phishing(1)", "Defacement(2)", "Malware(3)"]
    ))

# ─────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best_model(name, model, scaler, feature_names):
    joblib.dump({
        "model":         model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    name,
    }, "best_model.pkl")
    print(f"\n[SAVED] Best model saved to 'best_model.pkl'")

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  SENTINEL URL — Phase 2 Step 1: Model Comparison")
    print("  4-Class: Benign | Phishing | Defacement | Malware")
    print("="*60)

    urls, labels = load_dataset("encoded_dataset.csv")

    X = build_feature_matrix(urls)
    y = np.array(labels)

    feature_names = X.columns.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    results, (best_name, best_model) = train_and_compare(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("  FINAL COMPARISON TABLE")
    print("="*60)
    print(f"  {'Model':<25} {'Accuracy':>9} {'F1 Score':>9} {'ROC-AUC':>9}")
    print(f"  {'─'*55}")
    for name, r in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<25} {r['accuracy']:>9.4f} {r['f1_score']:>9.4f} {r['roc_auc']:>9.4f}{marker}")

    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print_confusion_matrix(best_model, X_test, y_test)

    if "Random Forest" in best_name:
        importances = best_model.feature_importances_
        feat_imp    = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\n  Top 5 Most Important Features:")
        for feat, imp in feat_imp[:5]:
            bar = "█" * int(imp * 100)
            print(f"  {feat:<25} {imp:.4f}  {bar}")

    save_best_model(best_name, best_model, scaler, feature_names)

    print("\n[DONE] Step 1 complete! Now move to Step 2 — charts & graphs.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
