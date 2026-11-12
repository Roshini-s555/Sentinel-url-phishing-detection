"""
Sentinel URL - Phase 2
Dataset Cleaning + Label Encoding (0,1,2,3)
Run: python clean_and_encode.py
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# STEP 1 - LOAD DATASET
# ─────────────────────────────────────────
print("="*55)
print("  SENTINEL URL - Dataset Cleaning & Encoding")
print("="*55)

df = pd.read_csv("cleaned_dataset.csv")
print(f"\n[STEP 1] Loaded dataset")
print(f"         Total rows    : {len(df):,}")
print(f"         Columns       : {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 2 - SHOW BEFORE ENCODING
# ─────────────────────────────────────────
print(f"\n[STEP 2] Labels BEFORE encoding:")
print(f"         {'Label':<15} {'Count':>8} {'Percent':>8}")
print(f"         {'─'*35}")
for cls, cnt in df['type'].value_counts().items():
    print(f"         {cls:<15} {cnt:>8,} {cnt/len(df)*100:>7.1f}%")

# ─────────────────────────────────────────
# STEP 3 - LABEL ENCODING
# ─────────────────────────────────────────
# Map text labels to numbers
label_map = {
    'benign'     : 0,   # ✅ Safe
    'phishing'   : 1,   # 🎣 Credential theft
    'defacement' : 2,   # 🖤 Hacked site
    'malware'    : 3,   # 💀 Silent infection
}

df['label'] = df['type'].map(label_map)

print(f"\n[STEP 3] Label Encoding Map:")
print(f"         benign      →  0  (Safe)")
print(f"         phishing    →  1  (Credential theft)")
print(f"         defacement  →  2  (Hacked site)")
print(f"         malware     →  3  (Silent infection)")

# ─────────────────────────────────────────
# STEP 4 - SHOW AFTER ENCODING
# ─────────────────────────────────────────
print(f"\n[STEP 4] Labels AFTER encoding:")
print(f"         {'Label':<15} {'Number':>8} {'Count':>8}")
print(f"         {'─'*35}")
for cls, num in label_map.items():
    cnt = len(df[df['label'] == num])
    print(f"         {cls:<15} {num:>8} {cnt:>8,}")

# ─────────────────────────────────────────
# STEP 5 - VERIFY NO NULLS IN LABEL
# ─────────────────────────────────────────
null_labels = df['label'].isnull().sum()
print(f"\n[STEP 5] Null labels after encoding: {null_labels}")
if null_labels > 0:
    print(f"         Dropping {null_labels} rows with unknown labels...")
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
else:
    df['label'] = df['label'].astype(int)
    print(f"         All labels encoded successfully! ✅")

# ─────────────────────────────────────────
# STEP 6 - SHOW SAMPLE ROWS
# ─────────────────────────────────────────
print(f"\n[STEP 6] Sample rows after encoding:")
print(f"\n  {'URL':<50} {'Type':<12} {'Label'}")
print(f"  {'─'*70}")
for cls in ['benign', 'phishing', 'defacement', 'malware']:
    row = df[df['type'] == cls].iloc[0]
    url_short = str(row['url'])[:48] + '..' if len(str(row['url'])) > 48 else str(row['url'])
    print(f"  {url_short:<50} {row['type']:<12} {int(row['label'])}")

# ─────────────────────────────────────────
# STEP 7 - SAVE ENCODED DATASET
# ─────────────────────────────────────────
# Save with all 3 columns: url, type, label
df[['url', 'type', 'label']].to_csv("encoded_dataset.csv", index=False)

print(f"\n[STEP 7] Saved encoded_dataset.csv")
print(f"         Columns: url | type | label")
print(f"         Total rows: {len(df):,}")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ENCODING COMPLETE!")
print(f"{'='*55}")
print(f"  0 = benign      ({len(df[df['label']==0]):,} URLs)")
print(f"  1 = phishing    ({len(df[df['label']==1]):,} URLs)")
print(f"  2 = defacement  ({len(df[df['label']==2]):,} URLs)")
print(f"  3 = malware     ({len(df[df['label']==3]):,} URLs)")
print(f"\n  Next step → run feature extraction on encoded_dataset.csv")
print(f"{'='*55}\n")
