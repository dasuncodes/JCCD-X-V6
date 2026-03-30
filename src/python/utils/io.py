from pathlib import Path

import numpy as np
import pandas as pd


def load_csv_pairs(path: str | Path, has_header: bool = False) -> pd.DataFrame:
    path = Path(path)
    if has_header:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "function_id_one" in df.columns:
            df = df.rename(
                columns={"function_id_one": "id1", "function_id_two": "id2"}
            )
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=["id1", "id2", "clone_type", "rare_token_ratio", "common_token_ratio"],
        )
    df["id1"] = df["id1"].astype(str).str.strip()
    df["id2"] = df["id2"].astype(str).str.strip()
    return df


def load_source_code(function_id: str, source_dir: str | Path) -> str | None:
    file_path = Path(source_dir) / f"{function_id}.java"
    if not file_path.exists():
        return None
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def validate_pair(id1: str, id2: str, source_dir: str | Path) -> bool:
    src1 = Path(source_dir) / f"{id1}.java"
    src2 = Path(source_dir) / f"{id2}.java"
    return src1.exists() and src2.exists()


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
