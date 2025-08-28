from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import _tree

API_URL = "https://www.catercow.com/cpi/v2/public/order_surveys"

API_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "dnt": "1",
    "origin": "https://cater-cow.s3.us-west-1.amazonaws.com",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://cater-cow.s3.us-west-1.amazonaws.com/",
    "sec-ch-ua": '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
}


def _fetch_page(client: httpx.Client, limit: int, offset: int) -> Dict[str, Any]:
    params = {
        "team_id": "8728D1C81AF4129A3D1E48683F3B59E47A692DC6",
        "limit": limit,
        "offset": offset,
    }
    r = client.get(API_URL, headers=API_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_records(limit: int = 100, max_pages: int = 100) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with httpx.Client() as client:
        offset = 0
        page = 0
        total_count = None
        while page < max_pages:
            data = _fetch_page(client, limit=limit, offset=offset)
            page_records = data.get("records", [])
            meta = data.get("meta", {})
            if total_count is None:
                total_count = meta.get("count")
            records.extend(page_records)
            if not page_records or len(page_records) < limit:
                break
            offset += limit
            page += 1
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        order = r.get("order", {}) or {}
        pkg = order.get("package", {}) or {}
        brand = order.get("brand", {}) or {}
        pricing = order.get("pricing", {}) or {}
        # Some fields
        start_date = order.get("start_date") or order.get("start_at")
        # Coerce price to float; price sometimes str
        price_raw = pkg.get("price")
        try:
            price = float(price_raw) if price_raw is not None else np.nan
        except Exception:
            price = np.nan

        # Prefer total_price from pricing when available
        def _to_float(x: Any) -> float:
            try:
                return float(x) if x is not None else np.nan
            except Exception:
                return np.nan

        total_price = _to_float(pricing.get("total_price"))
        customer_price = _to_float(pricing.get("customer_price"))

        row = {
            "survey_id": r.get("id"),
            "order_id": order.get("id") or r.get("order_id"),
            "headcount": order.get("headcount"),
            "start_date": start_date,
            "start_at": order.get("start_at"),
            "pickup": order.get("pickup"),
            "brand_id": brand.get("id") or order.get("brand_id"),
            "brand_name": brand.get("name"),
            "package_id": pkg.get("id"),
            "package_name": pkg.get("name"),
            "package_price": price,
            "total_price": total_price,
            "customer_price": customer_price,
            "presentation_type": pkg.get("presentation_type"),
            "set_up_type": pkg.get("set_up_type"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Parse dates and derive weekday (vectorized with coercion)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
    df["start_at_ts"] = pd.to_datetime(df["start_at"], errors="coerce")
    df["dow"] = pd.to_datetime(df["start_date"]).dt.dayofweek
    # Basic cleaning
    df = df.dropna(subset=["headcount", "package_price", "start_date"]).copy()
    df["headcount"] = df["headcount"].astype(int)
    df["package_price"] = df["package_price"].astype(float)
    # Use API-provided total price when present; otherwise fall back
    if "total_price" in df.columns:
        df["total_price"] = df["total_price"].astype(float)
        df["estimated_cost"] = np.where(
            df["total_price"].notna(),
            df["total_price"],
            df["headcount"] * df["package_price"],
        )
    else:
        df["estimated_cost"] = df["headcount"] * df["package_price"]
    return df


def compute_weekday_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("dow")
        .agg(
            avg_headcount=("headcount", "mean"),
            avg_price_pp=("package_price", "mean"),
            n_orders=("survey_id", "count"),
        )
        .reset_index()
    )
    return stats


def make_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    # Features: headcount, package_price, interaction, dow one-hot
    X = df.copy()
    X["hc_x_price"] = X["headcount"] * X["package_price"]
    numeric_features = ["headcount", "package_price", "hc_x_price"]
    categorical_features = ["dow"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    model = LinearRegression()
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    feature_names = numeric_features + categorical_features
    return pipe, feature_names


def train_model(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    # Derive interaction feature
    df = df.copy()
    df["hc_x_price"] = df["headcount"] * df["package_price"]
    # Time-aware split: last 14 unique dates as test if possible
    df = df.sort_values("start_date").reset_index(drop=True)
    unique_dates = sorted(df["start_date"].unique())
    holdout_days = min(14, max(1, int(len(unique_dates) * 0.2)))
    split_date = unique_dates[-holdout_days]
    train_df = df[df["start_date"] < split_date]
    test_df = df[df["start_date"] >= split_date]
    if train_df.empty or test_df.empty:
        # Fallback: 80/20 split by row
        split_idx = int(0.8 * len(df))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    pipe, feat_names = make_pipeline(cast(pd.DataFrame, train_df))
    X_train: pd.DataFrame = cast(pd.DataFrame, train_df)
    y_train = np.asarray(train_df["estimated_cost"], dtype=float)
    X_test: pd.DataFrame = cast(pd.DataFrame, test_df)
    y_test = np.asarray(test_df["estimated_cost"], dtype=float)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
        if len(y_test)
        else None,
        "mae": float(mean_absolute_error(y_test, y_pred)) if len(y_test) else None,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "split_date": str(split_date),
    }
    return pipe, metrics


# ---------- Daily aggregation + forecasting model ----------
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level rows into business-day totals and simple stats.

    Returns columns: date, dow, total_cost, avg_headcount, avg_price_pp, n_orders
    Weekends are retained initially; consumers can filter if needed.
    """
    daily = (
        (
            df.groupby("start_date").agg(
                total_cost=("estimated_cost", "sum"),
                avg_headcount=("headcount", "mean"),
                avg_price_pp=("package_price", "mean"),
                n_orders=("survey_id", "count"),
            )
        )
        .reset_index()
        .rename(columns={"start_date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])  # normalize to Timestamp
    daily["dow"] = daily["date"].dt.dayofweek
    daily = daily.set_index("date").sort_index().reset_index()
    return daily


def _add_calendar_features(d: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": d,
            "dow": d.dt.dayofweek,
            "day": d.dt.day,
            "weekofyear": d.dt.isocalendar().week.astype(int),
            "month": d.dt.month,
            "quarter": d.dt.quarter,
        }
    )
    out["trend"] = (out["date"] - out["date"].min()).dt.days
    return out


def build_daily_training_frame(
    daily: pd.DataFrame, business_days_only: bool = True
) -> pd.DataFrame:
    """Create supervised learning frame with lags and rolling stats.

    - If business_days_only=True, filter out weekends before creating lags so lag-1
      means previous business day.
    - Produces columns: target `y` and feature columns for modeling.
    """
    df = daily.copy()
    if business_days_only:
        df = df[df["dow"] < 5].copy()
    df = df.set_index("date").sort_index().reset_index()

    # Target
    df["y"] = df["total_cost"].astype(float)

    # Calendar features
    cal = _add_calendar_features(
        cast(pd.Series, df["date"])
    )  # keeps original date separately
    df = df.join(cal.add_prefix("cal_"))

    # Lags and rolling means on y
    df["y_lag1"] = df["y"].shift(1)
    df["y_lag7"] = df["y"].shift(5) if business_days_only else df["y"].shift(7)
    df["y_roll7"] = df["y"].rolling(window=7, min_periods=3).mean()
    df["y_roll28"] = df["y"].rolling(window=28, min_periods=7).mean()

    feat_cols = [
        "cal_dow",
        "cal_day",
        "cal_weekofyear",
        "cal_month",
        "cal_quarter",
        "cal_trend",
        "y_lag1",
        "y_lag7",
        "y_roll7",
        "y_roll28",
    ]
    # Keep only modeling features + cal_date and target
    frame = df.dropna(subset=feat_cols + ["y"]).reset_index(drop=True)
    # Ensure we don't leak raw 'date' column; keep 'cal_date' only for alignment
    cols = ["cal_date"] + feat_cols + ["y"]
    frame = frame.loc[:, cols]
    frame = cast(pd.DataFrame, frame)
    return frame


def train_daily_model(
    df: pd.DataFrame,
) -> Tuple[GradientBoostingRegressor, Dict[str, Any], pd.DataFrame]:
    """Train a simple tree-based regressor on business-day totals with lag features.

    Returns (model, metrics, history_df)
    history_df includes columns [date, y] and is used for recursive forecasting.
    """
    daily = aggregate_daily(df)
    frame = build_daily_training_frame(daily, business_days_only=True)
    if frame.empty or len(frame) < 30:
        raise RuntimeError("Not enough daily data to train time-series model")

    # Time-aware split: last 20% (min 14) business days for test
    n_total = len(frame)
    n_test = max(14, int(0.2 * n_total))
    split = n_total - n_test
    train, test = frame.iloc[:split], frame.iloc[split:]

    X_train = train.drop(columns=["y", "cal_date"])  # model can accept DataFrame
    y_train = np.asarray(train["y"], dtype=float)
    X_test = test.drop(columns=["y", "cal_date"]) if len(test) else None

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    metrics: Dict[str, Any]
    if X_test is not None and len(test):
        y_test_arr = np.asarray(test["y"], dtype=float)
        y_pred = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
            "mae": float(mean_absolute_error(y_test_arr, y_pred)),
            "n_train_days": int(len(train)),
            "n_test_days": int(len(test)),
        }
    else:
        metrics = {
            "rmse": None,
            "mae": None,
            "n_train_days": int(len(train)),
            "n_test_days": 0,
        }

    # Build history (date, y) for export/forecasting
    frame_with_dates = daily.merge(
        frame[["cal_date", "y"]], left_on="date", right_on="cal_date", how="left"
    )
    frame_with_dates = frame_with_dates.dropna(subset=["y"])  # keep only rows in frame
    frame_with_dates = frame_with_dates[["date", "y"]]
    frame_with_dates = frame_with_dates.sort_values("date")

    hist: pd.DataFrame = cast(pd.DataFrame, frame_with_dates.copy())
    hist = hist.reset_index(drop=True)
    return model, metrics, hist


def fetch_data_cli(args: Any) -> None:
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    records = fetch_all_records(limit=args.limit, max_pages=args.max_pages)
    # Save full JSON and JSONL for convenience
    json_path = outdir / "order_surveys_all.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"records": records}, f)
    save_jsonl(records, outdir / "order_surveys_all.jsonl")
    print(f"Saved {len(records)} records to {json_path}")


def train_cli(args: Any) -> None:
    rawdir: Path = args.raw
    raw_json = rawdir / "order_surveys_all.json"
    if not raw_json.exists():
        raise SystemExit(
            f"Raw data not found at {raw_json}. Run `uv run python main.py fetch` first."
        )
    with raw_json.open("r", encoding="utf-8") as f:
        records = json.load(f).get("records", [])

    df = flatten_records(records)
    args.processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.processed)

    weekday_stats = compute_weekday_stats(df)
    artifacts: Path = args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)
    weekday_stats.to_parquet(artifacts / "weekday_stats.parquet")

    # Order-level regression (kept for reference/compatibility)
    model, metrics = train_model(df)
    joblib.dump(model, artifacts / "model.joblib")

    # Daily time-series model for varying-by-date predictions
    try:
        daily_model, daily_metrics, daily_hist = train_daily_model(df)
        joblib.dump(daily_model, artifacts / "daily_model.joblib")
        daily_hist.to_parquet(artifacts / "daily_history.parquet")
        metrics["daily_model"] = daily_metrics
    except Exception as e:
        metrics["daily_model_error"] = str(e)

    with (artifacts / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Training complete:")
    print(json.dumps(metrics, indent=2))


def predict_next_cli(args: Any) -> None:
    artifacts: Path = args.artifacts
    model_path = artifacts / "model.joblib"
    weekday_path = artifacts / "weekday_stats.parquet"
    if not model_path.exists() or not weekday_path.exists():
        raise SystemExit(
            "Artifacts missing. Run `uv run python main.py train` first to build model."
        )
    model: Pipeline = joblib.load(model_path)
    stats = pd.read_parquet(weekday_path)

    # Determine tomorrow
    today = date.today()
    tomorrow = today + timedelta(days=1)
    dow = tomorrow.weekday()
    row = stats[stats["dow"] == dow]
    if row.empty:
        # Fallback to overall average
        avg_headcount = float(stats["avg_headcount"].mean())
        avg_price = float(stats["avg_price_pp"].mean())
    else:
        row_df: pd.DataFrame = cast(pd.DataFrame, row)
        avg_headcount = float(row_df["avg_headcount"].iloc[0])
        avg_price = float(row_df["avg_price_pp"].iloc[0])

    X = pd.DataFrame(
        [
            {
                "headcount": avg_headcount,
                "package_price": avg_price,
                "hc_x_price": avg_headcount * avg_price,
                "dow": dow,
            }
        ]
    )
    pred = float(model.predict(X)[0])
    result = {
        "tomorrow": str(tomorrow),
        "weekday": dow,
        "avg_headcount": avg_headcount,
        "avg_price_pp": avg_price,
        "predicted_cost": pred,
    }
    print(json.dumps(result, indent=2))


# ---------- Export for browser (edge) inference ----------
def _export_gbr_trees(model: GradientBoostingRegressor) -> Dict[str, Any]:
    trees: List[Dict[str, Any]] = []
    # estimators_ shape: (n_estimators, 1) for regression
    for i in range(model.estimators_.shape[0]):
        dt = model.estimators_[i, 0]
        t = dt.tree_
        nodes: List[Dict[str, Any]] = []
        for n in range(t.node_count):
            f = int(t.feature[n])  # -2 means leaf
            node = {
                "f": f,
                "t": float(t.threshold[n]) if f != _tree.TREE_UNDEFINED else None,
                "l": int(t.children_left[n]),
                "r": int(t.children_right[n]),
                "v": float(t.value[n][0][0]),
            }
            nodes.append(node)
        trees.append({"nodes": nodes})
    base_value: float
    try:
        # DummyRegressor stores [[value]]
        base_value = float(model.init_.constant_[0][0])  # type: ignore[attr-defined]
    except Exception:
        base_value = 0.0
    return {
        "base_value": base_value,
        "learning_rate": float(model.learning_rate),
        "trees": trees,
    }


def export_edge_cli(args: Any) -> None:
    artifacts: Path = args.artifacts
    model_path = artifacts / "daily_model.joblib"
    hist_path = artifacts / "daily_history.parquet"
    wday_path = artifacts / "weekday_stats.parquet"
    if not model_path.exists() or not hist_path.exists() or not wday_path.exists():
        raise SystemExit(
            "Missing artifacts. Run `uv run python main.py train` before exporting."
        )

    daily_model: GradientBoostingRegressor = joblib.load(model_path)
    model_json = _export_gbr_trees(daily_model)

    hist_df = pd.read_parquet(hist_path)
    hist_df = cast(pd.DataFrame, hist_df)
    hist_df["date"] = pd.to_datetime(hist_df["date"])  # ensure Timestamp
    trend0 = hist_df["date"].min().date().isoformat()
    history = {
        "y": [float(v) for v in hist_df["y"].tolist()],
        "trend0": trend0,
    }

    wday = pd.read_parquet(wday_path)
    wday = cast(pd.DataFrame, wday)
    weekday_stats: Dict[str, Any] = {}
    for _, row in wday.iterrows():
        key = str(int(row["dow"]))
        weekday_stats[key] = {
            "avg_headcount": float(row["avg_headcount"]),
            "avg_price_pp": float(row["avg_price_pp"]),
        }

    bundle = {
        "model": model_json,
        "history": history,
        "weekday_stats": weekday_stats,
    }

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f)
    print(f"Wrote edge bundle to {out_path}")

__all__ = [
    "fetch_data_cli",
    "predict_next_cli",
    "train_cli",
    "export_edge_cli",
]
