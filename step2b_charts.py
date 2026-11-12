"""
SentinelURL - Phase 2, Step 2B
Charts and Graphs: Confusion Matrix, ROC Curve, Feature Importance,
Model Comparison, Class Distribution, Per-Class Metrics
Run: python step2b_charts.py
Requirements: pip install matplotlib seaborn scikit-learn joblib pandas numpy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  SENTINEL URL - Step 2B: Charts & Graphs")
print("=" * 60)

# ─────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────
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
        entropy = -sum(p * math.log2(p) for p in prob if p > 0)
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

# ─────────────────────────────────────────
# 2. LOAD DATASET & MODEL
# ─────────────────────────────────────────
print("\n[STEP 1] Loading dataset and model...")
df = pd.read_csv("encoded_dataset.csv")
print(f"         Loaded {len(df):,} URLs")

data          = joblib.load("best_model.pkl")
model         = data["model"]
scaler        = data["scaler"]
feature_names = data["feature_names"]
print(f"         Model loaded: {data['model_name']}")

# ─────────────────────────────────────────
# 3. PREPARE FEATURES
# ─────────────────────────────────────────
print("\n[STEP 2] Extracting features (3-4 mins)...")
urls   = df["url"].astype(str).tolist()
labels = df["label"].tolist()

features_list = []
total = len(urls)
for i, url in enumerate(urls):
    features_list.append(extract_features(url))
    if (i+1) % 100000 == 0:
        print(f"         Progress: {i+1:,} / {total:,} ({(i+1)/total*100:.1f}%)")

X = pd.DataFrame(features_list)
y = np.array(labels)
X_scaled = scaler.transform(X)

_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

CLASS_NAMES = ["Benign", "Phishing", "Defacement", "Malware"]
COLORS      = ["#16a34a", "#dc2626", "#d97706", "#9333ea"]

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

# ─────────────────────────────────────────
# CHART 1: CONFUSION MATRIX
# ─────────────────────────────────────────
print("\n[CHART 1] Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax, linewidths=0.5,
    annot_kws={"size": 12, "weight": "bold"}
)
ax.set_title('Confusion Matrix — Random Forest\nSentinelURL Phase 2',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Actual Label',    fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig("chart1_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart1_confusion_matrix.png  OK")

# ─────────────────────────────────────────
# CHART 2: ROC CURVE
# ─────────────────────────────────────────
print("\n[CHART 2] ROC Curve...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_val     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f'{cls}  (AUC = {roc_val:.4f})')

ax.plot([0,1],[0,1], 'k--', lw=1.5, alpha=0.5, label='Random Guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate',  fontsize=12)
ax.set_title('ROC Curve — Random Forest\nSentinelURL Phase 2',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart2_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart2_roc_curve.png  OK")

# ─────────────────────────────────────────
# CHART 3: FEATURE IMPORTANCE
# ─────────────────────────────────────────
print("\n[CHART 3] Feature Importance...")
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': importances
}).sort_values('importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(feat_df['feature'], feat_df['importance'],
               color='#2563eb', edgecolor='white', height=0.6)
for bar, val in zip(bars, feat_df['importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color='#1e293b')
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Top 10 Feature Importance — Random Forest\nSentinelURL Phase 2',
             fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=10)
ax.set_xlim(0, max(feat_df['importance']) * 1.18)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("chart3_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart3_feature_importance.png  OK")

# ─────────────────────────────────────────
# CHART 4: MODEL COMPARISON
# ─────────────────────────────────────────
print("\n[CHART 4] Model Comparison...")
model_names = ['Random\nForest', 'XGBoost\n(GBM)', 'Logistic\nRegression']
accuracies  = [94.28, 91.21, 86.87]
f1_scores   = [94.22, 90.73, 84.94]
roc_aucs    = [98.51, 97.66, 92.35]
bar_colors  = ['#16a34a', '#2563eb', '#d97706']

x     = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)',  color=['#16a34acc','#2563ebcc','#d97706cc'])
b2 = ax.bar(x,          f1_scores,  width, label='F1 Score (%)', color=bar_colors)
b3 = ax.bar(x + width,  roc_aucs,   width, label='ROC-AUC (%)', color=['#16a34a88','#2563eb88','#d9770688'])

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Comparison — Accuracy, F1 & ROC-AUC\nSentinelURL Phase 2',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(80, 103)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("chart4_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart4_model_comparison.png  OK")

# ─────────────────────────────────────────
# CHART 5: CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\n[CHART 5] Class Distribution...")
class_counts = [427662, 93909, 94787, 23645]
explode      = (0.05, 0.05, 0.05, 0.10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

wedges, texts, autotexts = ax1.pie(
    class_counts, labels=CLASS_NAMES,
    colors=COLORS, autopct='%1.1f%%',
    startangle=90, explode=explode,
    pctdistance=0.82, textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title('Class Distribution\n(6,40,003 URLs)', fontsize=13, fontweight='bold')

bars2 = ax2.bar(CLASS_NAMES, class_counts, color=COLORS, edgecolor='white', width=0.5)
for bar, val in zip(bars2, class_counts):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3000,
             f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Number of URLs', fontsize=12)
ax2.set_title('URL Count per Class', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(class_counts) * 1.15)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Dataset Class Distribution — SentinelURL Phase 2',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("chart5_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart5_class_distribution.png  OK")

# ─────────────────────────────────────────
# CHART 6: PER-CLASS METRICS
# ─────────────────────────────────────────
print("\n[CHART 6] Per-Class Metrics...")
precision_vals = [0.96, 0.83, 0.96, 0.98]
recall_vals    = [0.97, 0.79, 0.99, 0.90]
f1_vals        = [0.96, 0.81, 0.97, 0.94]

x     = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#2563eb', alpha=0.85)
b2 = ax.bar(x,          recall_vals,    width, label='Recall',    color='#16a34a', alpha=0.85)
b3 = ax.bar(x + width,  f1_vals,        width, label='F1-Score',  color='#9333ea', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance — Random Forest\nSentinelURL Phase 2',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylim(0.7, 1.05)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("chart6_per_class_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("         Saved: chart6_per_class_metrics.png  OK")

# ─────────────────────────────────────────
# DONE
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2B COMPLETE! All 6 charts generated!")
print("="*60)
print("\n  Files saved in your url folder:")
print("  chart1_confusion_matrix.png")
print("  chart2_roc_curve.png")
print("  chart3_feature_importance.png")
print("  chart4_model_comparison.png")
print("  chart5_class_distribution.png")
print("  chart6_per_class_metrics.png")
print("\n  Add all 6 charts to your Phase 2 report!")
print("="*60)
