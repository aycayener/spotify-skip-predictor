from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

TZ = "Europe/Istanbul"
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/listens.parquet")

EXPECTED_COLS = [
    "ts","platform","ms_played","conn_country",
    "master_metadata_track_name","master_metadata_album_artist_name","master_metadata_album_album_name",
    "spotify_track_uri","episode_name","episode_show_name","spotify_episode_uri",
    "reason_start","reason_end","shuffle","skipped","offline","incognito_mode"
]

def load_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[EXPECTED_COLS]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df[df["ts"].notna()].copy()
    df["ts_local"] = df["ts"].dt.tz_convert(TZ)

    df["year"] = df["ts_local"].dt.year
    df["month"] = df["ts_local"].dt.month
    df["dayofweek"] = df["ts_local"].dt.dayofweek
    df["hour"] = df["ts_local"].dt.hour
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    def part_of_day(h: int) -> str:
        if pd.isna(h): return "unknown"
        if 5 <= h <= 10: return "morning"
        if 11 <= h <= 16: return "midday"
        if 17 <= h <= 22: return "evening"
        return "night"
    df["part_of_day"] = df["hour"].map(part_of_day)
    return df

def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    df["play_minutes"] = (pd.to_numeric(df["ms_played"], errors="coerce").fillna(0) / 60000.0).clip(lower=0)
    df["is_skip"] = df["skipped"].fillna(False).astype(bool).astype(int)
    df["is_shuffle"] = df["shuffle"].fillna(False).astype(bool).astype(int)
    df["is_offline"] = df["offline"].fillna(False).astype(bool).astype(int)

    df["content_type"] = np.where(df["spotify_track_uri"].notna(), "track",
                           np.where(df["spotify_episode_uri"].notna(), "episode", "unknown"))

    df["item_name"] = df["master_metadata_track_name"]
    df.loc[df["content_type"]=="episode", "item_name"] = df["episode_name"]

    df["creator_name"] = df["master_metadata_album_artist_name"]
    df.loc[df["content_type"]=="episode", "creator_name"] = df["episode_show_name"]

    df["is_noise"] = (df["play_minutes"] < (10/60)).astype(int)  # <10 sec
    return df

def add_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    dt_sec = df["ts"].diff().dt.total_seconds()
    new_session = (dt_sec.isna()) | (dt_sec > gap_minutes*60)
    df["session_id"] = new_session.cumsum().astype(int)
    df["session_pos"] = df.groupby("session_id").cumcount()
    df["session_size"] = df.groupby("session_id")["session_id"].transform("size")
    return df

def main() -> None:
    paths = sorted(RAW_DIR.glob("*.json"))
    if not paths:
        raise SystemExit("data/raw altında json bulunamadı.")
    frames = [load_json(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df = add_time_features(df)
    df = add_behavior_features(df)
    df = add_sessions(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] {OUT_PATH} yazıldı | rows={len(df):,} cols={df.shape[1]}")

if __name__ == "__main__":
    main()
