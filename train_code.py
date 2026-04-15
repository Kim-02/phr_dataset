import math
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 0) 상수 / 라벨 정의
# =========================================================
LABEL_NO_REST = 0
LABEL_WEAK_REST = 1
LABEL_STRONG_REST = 2

LABEL_NAME_MAP = {
    LABEL_NO_REST: "미휴식",
    LABEL_WEAK_REST: "약한휴식권고",
    LABEL_STRONG_REST: "강한휴식권고",
}

MODEL_FEATURES = [
    "hr_30s_avg",
    "heat_index",
    "hr_delta_from_baseline",
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "elderly_flag",
    "risk_factor_count",
]


# =========================================================
# 1) 유틸
# =========================================================
def calc_heat_index_c(temp_c: float, rh: float) -> float:
    t1 = temp_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
    t2 = math.atan(temp_c + rh)
    t3 = math.atan(rh - 1.67633)
    t4 = 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
    tw = t1 + t2 - t3 + t4 - 4.686035
    hi = (
        -0.2442
        + (0.55399 * tw)
        + (0.45535 * temp_c)
        - (0.0022 * tw ** 2)
        + (0.00278 * tw * temp_c)
        + 3.0
    )
    return round(float(hi), 2)


def encode_gender(gender_value: str) -> int:
    g = str(gender_value).strip().lower()
    if g == "m":
        return 1
    elif g == "f":
        return 0
    return -1


def normalize_subject_id(subject_id: str) -> str:
    s = str(subject_id).strip()
    if "_" in s:
        return s.split("_")[0]
    return s


# =========================================================
# 2) 문헌 기반 모델 라벨 생성
#    0 = 미휴식
#    1 = 약한휴식권고
#    2 = 강한휴식권고
# =========================================================
def make_model_label_literature_based(
    *,
    age: int,
    current_hr: float,
    baseline_hr: float,
    temp_c: float,
    humid: float,
    elderly_flag: int,
    heart_disease: int,
    hypertension: int,
    other_disease: int,
) -> int:
    heat_index = calc_heat_index_c(temp_c, humid)
    risk_factor_count = (
        int(elderly_flag)
        + int(heart_disease)
        + int(hypertension)
        + int(other_disease)
    )
    hr_delta = float(current_hr) - float(baseline_hr)
    hr_limit = 180 - int(age)
    hr_warn = 0.90 * hr_limit

    # 강한휴식권고
    if heat_index >= 39.4:
        return LABEL_STRONG_REST

    if current_hr >= hr_limit and heat_index >= 32.8:
        return LABEL_STRONG_REST

    if hr_delta >= 30 and heat_index >= 33.0:
        return LABEL_STRONG_REST

    if age >= 60 and risk_factor_count >= 2 and heat_index >= 32.8:
        return LABEL_STRONG_REST

    # 약한휴식권고
    if heat_index >= 32.8:
        return LABEL_WEAK_REST

    if current_hr >= hr_warn and heat_index >= 30.5:
        return LABEL_WEAK_REST

    if hr_delta >= 20 and heat_index >= 30.5:
        return LABEL_WEAK_REST

    if risk_factor_count >= 1 and heat_index >= 31.0:
        return LABEL_WEAK_REST

    return LABEL_NO_REST


# =========================================================
# 3) 데이터셋 빌더
# =========================================================
class WearableDatasetBuilder:
    def __init__(self, dataset_root: str, subject_info_path: str):
        self.dataset_root = Path(dataset_root)
        self.subject_info_path = Path(subject_info_path)

    def read_empatica_csv(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, header=None)

        first_value = raw.iloc[0, 0]
        second_value = raw.iloc[1, 0]

        try:
            start_time = float(first_value)
            timestamp_mode = "epoch"
        except ValueError:
            start_time = pd.to_datetime(first_value)
            timestamp_mode = "datetime"

        sample_rate = float(second_value)
        values = raw.iloc[2:].reset_index(drop=True)

        if values.shape[1] == 1:
            values.columns = ["value"]
        else:
            values.columns = [f"value_{i}" for i in range(values.shape[1])]

        for col in values.columns:
            values[col] = pd.to_numeric(values[col], errors="coerce")

        if timestamp_mode == "epoch":
            values["timestamp"] = start_time + np.arange(len(values)) / sample_rate
        else:
            values["timestamp"] = start_time + pd.to_timedelta(
                np.arange(len(values)) / sample_rate, unit="s"
            )

        return values

    def load_subject_info(self) -> pd.DataFrame:
        df = pd.read_csv(self.subject_info_path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def get_subject_profile(self, subject_info_df: pd.DataFrame, subject_id: str) -> Optional[dict]:
        subject_key = normalize_subject_id(subject_id)

        id_col = "info"
        gender_col = "gender"
        age_col = "age"
        height_col = "height (cm)"
        weight_col = "weight (kg)"

        if id_col not in subject_info_df.columns:
            raise ValueError(f"'info' 컬럼이 없습니다. 현재 컬럼: {list(subject_info_df.columns)}")

        row = subject_info_df[subject_info_df[id_col].astype(str).str.strip() == subject_key]
        if row.empty:
            return None

        row = row.iloc[0]

        age = pd.to_numeric(str(row[age_col]).strip(), errors="coerce")
        height_cm = pd.to_numeric(str(row[height_col]).strip(), errors="coerce")
        weight_kg = pd.to_numeric(str(row[weight_col]).strip(), errors="coerce")

        if pd.isna(age) or pd.isna(height_cm) or pd.isna(weight_kg):
            return None

        return {
            "age": int(age),
            "gender": encode_gender(row[gender_col]),
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
        }

    def assign_disease_flags(self, subject_id: str, session_type: str = "") -> Tuple[int, int, int]:
        numeric_part = "".join([ch for ch in subject_id if ch.isdigit()])
        base_seed = int(numeric_part) if numeric_part else 0
        seed = base_seed + sum(ord(c) for c in session_type)
        rng = np.random.default_rng(seed)

        heart_disease = int(rng.random() < 0.12)
        hypertension = int(rng.random() < 0.25)
        other_disease = int(rng.random() < 0.18)
        return heart_disease, hypertension, other_disease

    def generate_humidity_series(self, n_rows: int, session_type: str, seed: int):
        rng = np.random.default_rng(seed)

        if session_type == "STRESS":
            hum = rng.normal(loc=68, scale=10, size=n_rows)
        elif session_type == "AEROBIC":
            hum = rng.normal(loc=58, scale=12, size=n_rows)
        elif session_type == "ANAEROBIC":
            hum = rng.normal(loc=52, scale=9, size=n_rows)
        else:
            hum = rng.normal(loc=57, scale=11, size=n_rows)

        hum = np.clip(hum, 30, 90)
        return np.round(hum, 1)

    def build_session_dataset(
        self,
        session_dir: Path,
        subject_id: str,
        session_type: str,
        profile: dict,
    ) -> Optional[pd.DataFrame]:
        hr_path = session_dir / "HR.csv"
        temp_path = session_dir / "TEMP.csv"

        if not hr_path.exists() or not temp_path.exists():
            return None

        hr_df = self.read_empatica_csv(hr_path).rename(columns={"value": "hr"})
        temp_df = self.read_empatica_csv(temp_path).rename(columns={"value": "temp"})

        hr_df["hr"] = pd.to_numeric(hr_df["hr"], errors="coerce")
        temp_df["temp"] = pd.to_numeric(temp_df["temp"], errors="coerce")

        merged = pd.merge_asof(
            hr_df.sort_values("timestamp"),
            temp_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        merged = merged.dropna(subset=["hr", "temp"]).copy()

        base_time = merged["timestamp"].min()
        if pd.api.types.is_datetime64_any_dtype(merged["timestamp"]):
            time_diff_sec = (merged["timestamp"] - base_time).dt.total_seconds()
        else:
            time_diff_sec = merged["timestamp"] - base_time

        merged["window_id"] = (time_diff_sec // 30).astype(int)

        agg = merged.groupby("window_id").agg(
            window_start=("timestamp", "min"),
            hr_30s_avg=("hr", "mean"),
            temp_30s_avg=("temp", "mean"),
        ).reset_index(drop=True)

        if len(agg) == 0:
            return None

        agg["subject_id"] = subject_id
        agg["session_type"] = session_type
        agg["age"] = profile["age"]
        agg["gender"] = profile["gender"]
        agg["height_cm"] = profile["height_cm"]
        agg["weight_kg"] = profile["weight_kg"]
        agg["elderly_flag"] = (agg["age"] >= 60).astype(int)

        humidity_seed = sum(ord(c) for c in (subject_id + session_type))
        agg["hum_30s_avg"] = self.generate_humidity_series(len(agg), session_type, humidity_seed)

        heart_disease, hypertension, other_disease = self.assign_disease_flags(subject_id, session_type)
        agg["heart_disease"] = heart_disease
        agg["hypertension"] = hypertension
        agg["other_disease"] = other_disease

        agg["risk_factor_count"] = (
            agg["elderly_flag"]
            + agg["heart_disease"]
            + agg["hypertension"]
            + agg["other_disease"]
        )

        agg["heat_index"] = [
            calc_heat_index_c(t, h)
            for t, h in zip(agg["temp_30s_avg"].values, agg["hum_30s_avg"].values)
        ]

        baseline_hr = float(agg["hr_30s_avg"].iloc[0])
        agg["baseline_hr"] = baseline_hr
        agg["hr_delta_from_baseline"] = agg["hr_30s_avg"] - baseline_hr

        agg["model_label"] = agg.apply(
            lambda row: make_model_label_literature_based(
                age=int(row["age"]),
                current_hr=float(row["hr_30s_avg"]),
                baseline_hr=float(row["baseline_hr"]),
                temp_c=float(row["temp_30s_avg"]),
                humid=float(row["hum_30s_avg"]),
                elderly_flag=int(row["elderly_flag"]),
                heart_disease=int(row["heart_disease"]),
                hypertension=int(row["hypertension"]),
                other_disease=int(row["other_disease"]),
            ),
            axis=1,
        )

        return agg[
            [
                "subject_id",
                "session_type",
                "window_start",
                "hr_30s_avg",
                "temp_30s_avg",
                "hum_30s_avg",
                "heat_index",
                "baseline_hr",
                "hr_delta_from_baseline",
                "age",
                "gender",
                "height_cm",
                "weight_kg",
                "elderly_flag",
                "heart_disease",
                "hypertension",
                "other_disease",
                "risk_factor_count",
                "model_label",
            ]
        ]

    def build_all_dataset(self) -> pd.DataFrame:
        subject_info_df = self.load_subject_info()

        session_types = ["AEROBIC", "ANAEROBIC", "STRESS"]
        all_dfs = []

        for session_type in session_types:
            session_root = self.dataset_root / session_type
            if not session_root.exists():
                continue

            for subject_dir in sorted(session_root.iterdir()):
                if not subject_dir.is_dir():
                    continue

                subject_id = subject_dir.name
                profile = self.get_subject_profile(subject_info_df, subject_id)

                if profile is None:
                    print(f"[SKIP] subject-info 누락/비정상: {subject_id}")
                    continue

                df = self.build_session_dataset(
                    session_dir=subject_dir,
                    subject_id=subject_id,
                    session_type=session_type,
                    profile=profile,
                )

                if df is not None and len(df) > 0:
                    all_dfs.append(df)
                    print(f"[OK] {session_type}/{subject_id} -> {len(df)} rows")

        if not all_dfs:
            raise ValueError("생성된 데이터가 없습니다.")

        return pd.concat(all_dfs, ignore_index=True)


# =========================================================
# 4) 학습기
# =========================================================
class RestRecommendationTrainer:
    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or MODEL_FEATURES
        self.model: Optional[Pipeline] = None

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
        X = df[self.feature_cols].copy()
        y = df["model_label"].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight=class_weight_dict,
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        return {
            "feature_cols": self.feature_cols,
            "confusion_matrix": confusion_matrix(y_test, pred),
            "classification_report": classification_report(
                y_test,
                pred,
                labels=[0, 1, 2],
                target_names=[
                    LABEL_NAME_MAP[0],
                    LABEL_NAME_MAP[1],
                    LABEL_NAME_MAP[2],
                ],
                digits=4,
                zero_division=0,
            ),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "weighted_f1": f1_score(y_test, pred, average="weighted"),
        }

    def save(self, model_path: str):
        if self.model is None:
            raise ValueError("학습된 모델이 없습니다.")
        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "label_name_map": LABEL_NAME_MAP,
        }
        joblib.dump(payload, model_path)


# =========================================================
# 5) 실행부
# =========================================================
if __name__ == "__main__":
    ROOT = Path("./wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1")
    DATASET_ROOT = ROOT / "Wearable_Dataset"
    SUBJECT_INFO_PATH = ROOT / "subject-info.csv"

    builder = WearableDatasetBuilder(
        dataset_root=str(DATASET_ROOT),
        subject_info_path=str(SUBJECT_INFO_PATH),
    )

    df = builder.build_all_dataset()
    df.to_csv("rest_model_dataset.csv", index=False)
    print("\n[DATASET SHAPE]")
    print(df.shape)

    print("\n[LABEL COUNTS]")
    print(df["model_label"].value_counts().sort_index())

    trainer = RestRecommendationTrainer()
    metrics = trainer.train(df)

    print("\n[FEATURE COLS]")
    print(metrics["feature_cols"])

    print("\n[CONFUSION MATRIX]")
    print(metrics["confusion_matrix"])

    print("\n[CLASSIFICATION REPORT]")
    print(metrics["classification_report"])

    print("\n[MACRO F1]")
    print(metrics["macro_f1"])

    print("\n[WEIGHTED F1]")
    print(metrics["weighted_f1"])

    trainer.save("rest_recommendation_model.pkl")
    print("\n모델 저장 완료: rest_recommendation_model.pkl")