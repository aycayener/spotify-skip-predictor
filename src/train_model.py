from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA_PATH = Path("data/processed/listens.parquet")
MODEL_PATH = Path("models/skip_model.pkl")
META_PATH = Path("models/skip_model_meta.json")

def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Önce: python src/data_prep.py")

    df = pd.read_parquet(DATA_PATH)
    df = df[df["content_type"].isin(["track", "episode"])].copy()
    df = df[df["is_noise"] == 0].copy()

    y = df["is_skip"].astype(int)

    num_cols = ["hour", "dayofweek", "play_minutes", "session_pos", "session_size", "is_shuffle", "is_offline"]
    cat_cols = ["part_of_day", "platform", "reason_start", "reason_end", "conn_country", "content_type"]

    X = df[num_cols + cat_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in cat_cols:
        X[c] = X[c].fillna("unknown").astype(str)

    df_sorted = df.sort_values("ts").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 'passthrough' string'i yerine standart StandardScaler kullanarak transformer hatasını önlüyoruz
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Classification report ===")
    print(classification_report(y_test, pred, digits=3))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    meta = {"num_cols": num_cols, "cat_cols": cat_cols}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Model başarıyla güncellendi ve kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    main()