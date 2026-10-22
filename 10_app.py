"""FraudShield AI - Transaction Risk Analysis Dashboard.

This Streamlit application is an inference-only layer around the model artifact
created by ``08_save_model.py``. It never trains or modifies a model, and it
never sends data over the network.
"""

from __future__ import annotations

import html as html_lib
import math
import textwrap
from pathlib import Path

import streamlit as st

try:
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as error:  # pragma: no cover - environment guard
    st.set_page_config(page_title="FraudShield AI", layout="wide")
    st.error(
        f"A required Python package is missing: {error.name}. "
        "Install the project dependencies with "
        "`pip install -r requirements.txt` and restart the application."
    )
    st.stop()


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "fraudshield_random_forest.joblib"
DATASET_PATH = PROJECT_DIR / "creditcard.csv"
# Small committed preview used when the complete 144 MB CSV is absent (for
# example on Streamlit Community Cloud). It is a sample, not the full dataset.
SAMPLE_DATASET_PATH = PROJECT_DIR / "data" / "sample_transactions.csv"
TARGET_COLUMN = "Class"

NAV_ITEMS = [
    "Overview",
    "Risk Analysis",
    "Dataset Explorer",
    "Model Performance",
    "Model Details",
]

# Historical threshold used by 07_final_evaluation.py for the verified test
# metrics below. Live predictions use the threshold stored in the loaded
# artifact, which may differ from this evaluation threshold.
EVALUATION_THRESHOLD = 0.20

# Verified results from 07_final_evaluation.py on the held-out test split.
# They are constants because the dashboard displays evaluation results; it does
# not retrain the model or recompute metrics from the test labels at runtime.
TEST_RESULTS = {
    "threshold": EVALUATION_THRESHOLD,
    "precision": 0.8261,
    "recall": 0.8028,
    "f1": 0.8143,
    "average_precision": 0.7967,
    "tn": 42_476,
    "fp": 12,
    "fn": 14,
    "tp": 57,
    "transactions": 42_559,
    "actual_fraud": 71,
    "actual_legitimate": 42_488,
}

METRIC_DEFINITIONS = [
    (
        "Precision",
        "Of every transaction the model flagged as fraud, how many were actually fraudulent?",
        "57 true positives / 69 fraud alerts = 82.61%",
    ),
    (
        "Recall",
        "Of all actual fraud cases, how many did the model successfully catch?",
        "57 detected / 71 actual frauds = 80.28%",
    ),
    (
        "F1-score",
        "A balance between precision and recall, useful when both matter.",
        "2 x P x R / (P + R) = 0.8143",
    ),
    (
        "Average Precision",
        "A threshold-independent summary of ranking quality across the precision-recall curve.",
        "0.7967 on the held-out test set",
    ),
]


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _validate_artifact(artifact) -> dict:
    """Validate the saved bundle before exposing it to the UI."""
    if not isinstance(artifact, dict):
        raise ValueError("The saved artifact is not a dictionary.")

    required_keys = {"model", "threshold", "feature_names"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            "The saved artifact is missing required keys: "
            + ", ".join(sorted(missing_keys))
            + "."
        )

    model = artifact["model"]
    feature_names = list(artifact["feature_names"])

    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise ValueError("The saved object does not look like a fitted classifier.")

    try:
        threshold = float(artifact["threshold"])
    except (TypeError, ValueError) as error:
        raise ValueError("The saved decision threshold is not numeric.") from error

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The saved decision threshold must be between 0 and 1.")

    if len(feature_names) != 30:
        raise ValueError(
            f"Expected 30 model features, found {len(feature_names)}."
        )
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("The saved feature list contains duplicate names.")
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError("The saved feature list contains an invalid name.")

    model_feature_count = getattr(model, "n_features_in_", None)
    if model_feature_count is not None and model_feature_count != len(feature_names):
        raise ValueError(
            "The model and the saved feature list disagree on the number of inputs "
            f"({model_feature_count} vs {len(feature_names)})."
        )

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        expected_features = list(model_feature_names)
        if expected_features != feature_names:
            mismatch = next(
                (
                    index
                    for index, (expected, saved) in enumerate(
                        zip(expected_features, feature_names)
                    )
                    if expected != saved
                ),
                min(len(expected_features), len(feature_names)),
            )
            raise ValueError(
                "The model's expected feature names or order do not match the saved "
                f"feature list (first mismatch at position {mismatch + 1}). "
                "Refusing to use an incompatible artifact."
            )

    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError(
            "The saved model does not expose classes_. FraudShield AI cannot verify "
            "that class label 1 corresponds to fraud."
        )
    classes = list(model_classes)
    if classes != [0, 1]:
        raise ValueError(
            "The saved model's classes_ must be exactly [0, 1] so that the fraud "
            f"probability corresponds to class label 1. Found: {classes}."
        )

    return {
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
    }


@st.cache_resource(show_spinner="Loading the saved model artifact...")
def load_model_artifact(path_string: str) -> dict:
    """Load the model bundle once per server process."""
    artifact = joblib.load(path_string)
    return _validate_artifact(artifact)


@st.cache_data(show_spinner="Reading the local transaction dataset...", max_entries=1)
def load_dataset(path_string: str) -> tuple[pd.DataFrame, dict]:
    """Read the dataset once and return it with a compact summary.

    The cache prevents a full 150 MB CSV read on every widget interaction.
    V1-V28 are stored as float32 to keep memory use reasonable.
    """
    dtypes = {f"V{index}": "float32" for index in range(1, 29)}
    dtypes.update({"Time": "int32", "Amount": "float32", "Class": "int8"})
    expected_columns = ["Time"] + [f"V{index}" for index in range(1, 29)] + [
        "Amount",
        "Class",
    ]

    try:
        data = pd.read_csv(path_string, dtype=dtypes)
    except pd.errors.EmptyDataError:
        # A zero-byte CSV has no header to parse. Treat it as a zero-row table
        # with the expected schema so callers can show a clear guard message.
        data = pd.DataFrame(columns=expected_columns)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"The dataset does not contain the required '{TARGET_COLUMN}' column."
        )

    class_counts = data[TARGET_COLUMN].value_counts()
    fraud_count = int(class_counts.get(1, 0))
    legitimate_count = int(class_counts.get(0, 0))
    total_rows = int(len(data))
    duplicate_rows = int(data.duplicated().sum())
    missing_total = int(data.isna().sum().sum())

    summary = {
        "rows": total_rows,
        "columns": int(data.shape[1]),
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_rate": (fraud_count / total_rows) if total_rows else 0.0,
        "duplicate_rows": duplicate_rows,
        "deduplicated_rows": total_rows - duplicate_rows,
        "missing_total": missing_total,
        "memory_mb": float(data.memory_usage(deep=True).sum() / (1024**2)),
    }
    return data, summary


def resolve_dataset(
    full_path: str | Path = DATASET_PATH,
    sample_path: str | Path = SAMPLE_DATASET_PATH,
) -> dict:
    """Choose the dataset the dashboard should read.

    The complete local ``creditcard.csv`` always wins. When it is absent, the
    committed sample preview is used instead so deployment-specific demos still
    work. Both paths are project-relative, not absolute machine-specific paths.
    """
    full = Path(full_path)
    sample = Path(sample_path)

    if full.is_file():
        return {
            "kind": "full",
            "path": full,
            "label": "creditcard.csv",
            "title": "Complete local dataset",
            "badge": "Full dataset",
            "is_sample": False,
        }
    if sample.is_file():
        return {
            "kind": "sample",
            "path": sample,
            "label": "data/sample_transactions.csv",
            "title": "Bundled sample preview",
            "badge": "Sample preview",
            "is_sample": True,
        }
    return {
        "kind": "missing",
        "path": None,
        "label": "creditcard.csv",
        "title": "Dataset unavailable",
        "badge": "Unavailable",
        "is_sample": False,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _fraud_probability_column(model) -> int:
    """Return the predict_proba column that corresponds to fraud label 1."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError(
            "The model does not expose classes_. Cannot identify the fraud "
            "probability column reliably."
        )
    class_labels = list(classes)
    if 1 not in class_labels:
        raise ValueError("The model's classes_ do not include fraud label 1.")
    return class_labels.index(1)


def model_input_values(row, feature_names: list[str]) -> dict:
    """Select exactly the model inputs, in the artifact's saved order.

    Dataset columns such as ``Class`` and any other extra fields are ignored.
    This keeps ground truth out of inference even when the source is a full
    dataset row.
    """
    missing = [name for name in feature_names if name not in row]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    try:
        return {name: float(row[name]) for name in feature_names}
    except (TypeError, ValueError) as error:
        raise ValueError("Every feature must be a valid numeric value.") from error


def predict_transaction(
    model,
    feature_names: list[str],
    threshold: float,
    values: dict,
) -> tuple[float, int]:
    """Run inference for exactly one transaction.

    A DataFrame is built with ``columns=feature_names`` so the model receives
    the same feature names and order that were saved with the artifact.
    """
    row = model_input_values(values, feature_names)
    frame = pd.DataFrame([row], columns=feature_names)
    frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)

    if frame.isna().any(axis=None):
        raise ValueError("Every feature must be a valid numeric value.")

    values_array = frame.to_numpy(dtype=float)
    if not np.isfinite(values_array).all():
        raise ValueError("Features must be finite numbers (no NaN or infinity).")

    fraud_column = _fraud_probability_column(model)
    probabilities = model.predict_proba(frame)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError("The model returned an unexpected probability shape.")
    if fraud_column >= probabilities.shape[1]:
        raise ValueError("The model returned fewer probability columns than expected.")

    probability = float(probabilities[0, fraud_column])
    prediction = int(probability >= threshold)
    return probability, prediction


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def html_block(markup: str) -> None:
    """Render trusted HTML through Streamlit.

    Fragments are normalized so every line starts at column zero. Without this,
    interpolated multi-line fragments can be interpreted by Markdown as indented
    code blocks instead of HTML.
    """
    cleaned = textwrap.dedent(markup).strip()
    cleaned = "\n".join(
        line.strip() for line in cleaned.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value))


def fmt_probability(probability: float) -> str:
    percentage = probability * 100.0
    if 0.0 < percentage < 0.01:
        return "&lt;0.01%"
    return f"{percentage:.2f}%"


def artifact_threshold_text(artifact: dict) -> str:
    """Threshold used for live predictions from the loaded artifact."""
    return f"{float(artifact['threshold']):.2f}"


def evaluation_threshold_text() -> str:
    """Historical threshold associated with the verified test metrics."""
    return f"{float(TEST_RESULTS['threshold']):.2f}"


def threshold_mismatch_message(artifact: dict) -> str | None:
    """Explain when the live artifact threshold differs from the evaluation one.

    The historical metrics are never recalculated. This message makes the
    distinction explicit instead of implying that the metrics track the artifact.
    """
    live_threshold = float(artifact["threshold"])
    evaluation_threshold = float(TEST_RESULTS["threshold"])
    if math.isclose(live_threshold, evaluation_threshold, rel_tol=0.0, abs_tol=1e-9):
        return None
    return (
        f"The loaded model artifact uses a decision threshold of {live_threshold:.2f}, "
        f"while the verified test metrics were evaluated at {evaluation_threshold:.2f}. "
        "Those historical metrics are shown exactly as recorded and are not "
        "recalculated for the artifact threshold. Live predictions use "
        f"{live_threshold:.2f}."
    )


def render_threshold_mismatch_warning(artifact: dict) -> None:
    message = threshold_mismatch_message(artifact)
    if message:
        st.warning(message)


def inject_css() -> None:
    """Apply the dark navy fintech theme and reusable component styles."""
    st.markdown(
        """
        <style>
        :root {
            --fs-bg: #060c18;
            --fs-panel: rgba(13, 25, 44, 0.88);
            --fs-panel-strong: #0e1b30;
            --fs-border: rgba(148, 163, 184, 0.14);
            --fs-text: #e8f0fc;
            --fs-muted: #93a7c2;
            --fs-cyan: #22d3ee;
            --fs-blue: #60a5fa;
            --fs-teal: #2dd4bf;
            --fs-rose: #fb7185;
            --fs-amber: #fbbf24;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "IBM Plex Sans", -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% -8%, rgba(34, 211, 238, 0.10), transparent 34%),
                radial-gradient(circle at 88% 2%, rgba(59, 130, 246, 0.10), transparent 30%),
                linear-gradient(180deg, #07101e 0%, #060c18 55%, #050a13 100%);
            color: var(--fs-text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        [data-testid="stAppDeployButton"] {
            display: none;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, h5 {
            color: var(--fs-text);
            letter-spacing: -0.02em;
        }

        p {
            color: #b7c7dd;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08111f 0%, #07101c 100%);
            border-right: 1px solid var(--fs-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        [data-testid="stSidebar"] .stButton {
            animation: fs-side-in 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        [data-testid="stSidebar"] .stButton > button {
            position: relative;
            overflow: hidden;
            width: 100%;
            justify-content: flex-start;
            background: transparent;
            border: 1px solid transparent;
            color: #9fb2cb;
            border-radius: 10px;
            padding: 0.58rem 0.8rem;
            font-weight: 600;
            transition:
                transform 0.22s cubic-bezier(0.22, 1, 0.36, 1),
                background 0.22s ease,
                border-color 0.22s ease,
                color 0.22s ease,
                box-shadow 0.28s ease;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(34, 211, 238, 0.075);
            border-color: rgba(34, 211, 238, 0.20);
            color: #e6fbff;
            transform: translateX(4px);
        }

        [data-testid="stSidebar"] .stButton > button:active {
            transform: translateX(2px) scale(0.99);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background:
                linear-gradient(
                    100deg,
                    rgba(34, 211, 238, 0.17) 0%,
                    rgba(59, 130, 246, 0.09) 55%,
                    rgba(34, 211, 238, 0.13) 100%
                );
            background-size: 180% 100%;
            border-color: rgba(34, 211, 238, 0.34);
            color: #e9fdff;
            transform: translateX(2px);
            animation: fs-nav-shift 5.5s ease-in-out infinite,
                fs-nav-glow 3.2s ease-in-out infinite;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
            content: "";
            position: absolute;
            top: 16%;
            bottom: 16%;
            left: 0;
            width: 3px;
            border-radius: 0 3px 3px 0;
            background: linear-gradient(180deg, #7dd3fc, #22d3ee 55%, #60a5fa);
            box-shadow: 0 0 12px rgba(34, 211, 238, 0.72);
            animation: fs-nav-bar 2.6s ease-in-out infinite;
        }

        .fs-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.2rem 0.15rem 1.35rem;
            border-bottom: 1px solid var(--fs-border);
            margin-bottom: 1.4rem;
            animation: fs-side-in 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        .fs-mark {
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            display: grid;
            place-items: center;
            color: var(--fs-cyan);
            border-radius: 12px;
            background: linear-gradient(145deg, rgba(34, 211, 238, 0.17), rgba(59, 130, 246, 0.06));
            border: 1px solid rgba(34, 211, 238, 0.28);
            box-shadow: 0 0 22px rgba(34, 211, 238, 0.10);
            animation: fs-mark-float 5s ease-in-out infinite;
        }

        .fs-mark svg {
            width: 24px;
            height: 24px;
        }

        .fs-brand-name {
            color: #f2f8ff;
            font-size: 1.02rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .fs-brand-name span {
            color: var(--fs-cyan);
        }

        .fs-brand-tag {
            color: #7f94af;
            font-size: 0.66rem;
            letter-spacing: 0.06em;
            line-height: 1.35;
            margin-top: 0.18rem;
        }

        .fs-nav-label {
            color: #657b98;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0.2rem 0 0.55rem 0.2rem;
            animation: fs-side-in 0.46s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        .fs-side-meta {
            margin-top: 1.6rem;
            padding-top: 1.1rem;
            border-top: 1px solid var(--fs-border);
            animation: fs-side-in 0.58s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        .fs-side-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            font-size: 0.75rem;
            color: #7f94af;
            padding: 0.22rem 0;
        }

        .fs-side-row b {
            color: #d9e8fa;
            font-weight: 600;
            text-align: right;
        }

        .fs-side-status {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.74rem;
            color: #9fb2cb;
            margin-top: 0.7rem;
        }

        .fs-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }

        .fs-dot-ok {
            background: var(--fs-teal);
            box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.10);
            animation: fs-status-pulse 2.8s ease-in-out infinite;
        }

        .fs-dot-bad {
            background: var(--fs-rose);
            box-shadow: 0 0 0 4px rgba(251, 113, 133, 0.10);
        }

        .fs-side-note {
            margin-top: 1.15rem;
            padding: 0.8rem 0.9rem;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(148, 163, 184, 0.04);
            color: #758aa5;
            font-size: 0.7rem;
            line-height: 1.55;
        }

        .fs-hero {
            padding: 0.6rem 0 1.2rem;
        }

        .fs-eyebrow {
            color: var(--fs-cyan);
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.19em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .fs-hero h1 {
            margin: 0 0 0.7rem;
            font-size: clamp(2.2rem, 4.4vw, 3.5rem);
            line-height: 1.03;
            font-weight: 760;
            letter-spacing: -0.045em;
            color: #f4f9ff;
        }

        .fs-gradient {
            background: linear-gradient(90deg, #7dd3fc 0%, #22d3ee 52%, #60a5fa 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .fs-hero p {
            color: #9db0c9;
            font-size: 1rem;
            line-height: 1.65;
            max-width: 68ch;
            margin: 0;
        }

        .fs-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1.2rem;
        }

        .fs-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(34, 211, 238, 0.18);
            background: rgba(34, 211, 238, 0.055);
            color: #a9e8f3;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .fs-page-title {
            margin: 0 0 0.45rem;
            font-size: clamp(1.9rem, 3.4vw, 2.7rem);
            line-height: 1.08;
            font-weight: 750;
            letter-spacing: -0.035em;
            color: #f4f9ff;
        }

        .fs-lead {
            color: #93a7c2;
            font-size: 0.95rem;
            line-height: 1.65;
            max-width: 78ch;
            margin: 0;
        }

        .fs-notice {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.9rem 1.05rem;
            margin: 1.15rem 0 0.3rem;
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 12px;
            background: rgba(59, 130, 246, 0.055);
            color: #9db4d0;
            font-size: 0.8rem;
            line-height: 1.6;
        }

        .fs-notice b {
            color: #cfe2fb;
        }

        .fs-notice-mark {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--fs-blue);
            margin-top: 0.4rem;
            flex: 0 0 7px;
            box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.11);
        }

        .fs-section-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1.2rem;
            margin: 2.1rem 0 0.95rem;
        }

        .fs-section-title {
            color: #ecf5ff;
            font-size: 1.06rem;
            font-weight: 720;
            letter-spacing: -0.015em;
        }

        .fs-section-sub {
            color: #7f94af;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .fs-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            white-space: nowrap;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(45, 212, 191, 0.22);
            background: rgba(45, 212, 191, 0.06);
            color: #9de9df;
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .fs-badge-dot {
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--fs-teal);
        }

        .fs-kpi {
            position: relative;
            min-height: 132px;
            overflow: hidden;
            padding: 1.08rem 1.15rem 1rem;
            border-radius: 15px;
            border: 1px solid var(--fs-border);
            background: linear-gradient(155deg, rgba(16, 31, 54, 0.92), rgba(9, 18, 33, 0.88));
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.13);
            animation: fs-rise 0.35s ease both;
        }

        .fs-kpi::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--fs-cyan), transparent 78%);
        }

        .fs-kpi.blue::before {
            background: linear-gradient(90deg, var(--fs-blue), transparent 78%);
        }

        .fs-kpi.teal::before {
            background: linear-gradient(90deg, var(--fs-teal), transparent 78%);
        }

        .fs-kpi.amber::before {
            background: linear-gradient(90deg, var(--fs-amber), transparent 78%);
        }

        .fs-kpi-label {
            color: #8297b2;
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .fs-kpi-value {
            color: #f2f9ff;
            font-size: 1.85rem;
            font-weight: 740;
            letter-spacing: -0.035em;
            line-height: 1.15;
            margin-top: 0.55rem;
        }

        .fs-kpi-value small {
            color: #89a0bb;
            font-size: 0.86rem;
            font-weight: 560;
            letter-spacing: 0;
        }

        .fs-kpi-foot {
            color: #7f94af;
            font-size: 0.72rem;
            line-height: 1.45;
            margin-top: 0.48rem;
        }

        .fs-panel {
            padding: 1.15rem 1.25rem;
            border-radius: 15px;
            border: 1px solid var(--fs-border);
            background: linear-gradient(160deg, rgba(14, 27, 48, 0.90), rgba(9, 18, 33, 0.88));
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
            animation: fs-rise 0.35s ease both;
        }

        .fs-panel-title {
            color: #dceafb;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .fs-muted {
            color: #8499b4;
            font-size: 0.76rem;
            line-height: 1.55;
        }

        .fs-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.4rem 0 0.2rem;
        }

        .fs-strip-item {
            padding: 0.78rem 0.9rem;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(148, 163, 184, 0.035);
        }

        .fs-strip-item span {
            display: block;
            color: #7187a3;
            font-size: 0.63rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .fs-strip-item b {
            display: block;
            color: #e5f0fd;
            font-size: 0.92rem;
            font-weight: 650;
            margin-top: 0.28rem;
        }

        .fs-cm {
            display: grid;
            grid-template-columns: 1.05fr 1fr 1fr;
            gap: 7px;
        }

        .fs-cm-axis {
            color: #778ca7;
            font-size: 0.63rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            text-align: center;
            padding: 0.2rem 0 0.35rem;
            line-height: 1.35;
        }

        .fs-cm-axis-y {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            text-align: left;
            padding-left: 0.1rem;
        }

        .fs-cm-cell {
            min-height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.11);
            background: rgba(148, 163, 184, 0.035);
            text-align: center;
            padding: 0.5rem;
        }

        .fs-cm-cell b {
            font-size: 1.28rem;
            font-weight: 720;
            letter-spacing: -0.03em;
            color: #eaf4ff;
        }

        .fs-cm-cell span {
            color: #8196b1;
            font-size: 0.62rem;
            line-height: 1.35;
            margin-top: 0.25rem;
        }

        .fs-cm-cell.good {
            background: rgba(45, 212, 191, 0.075);
            border-color: rgba(45, 212, 191, 0.20);
        }

        .fs-cm-cell.good b {
            color: #8ce9dd;
        }

        .fs-cm-cell.blue {
            background: rgba(96, 165, 250, 0.075);
            border-color: rgba(96, 165, 250, 0.20);
        }

        .fs-cm-cell.blue b {
            color: #9cc8ff;
        }

        .fs-cm-cell.warn {
            background: rgba(251, 191, 36, 0.065);
            border-color: rgba(251, 191, 36, 0.18);
        }

        .fs-cm-cell.warn b {
            color: #f8cf72;
        }

        .fs-cm-cell.bad {
            background: rgba(251, 113, 133, 0.065);
            border-color: rgba(251, 113, 133, 0.18);
        }

        .fs-cm-cell.bad b {
            color: #fda0ad;
        }

        .fs-def-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .fs-def {
            padding: 0.9rem 1rem;
            border-radius: 11px;
            border: 1px solid rgba(148, 163, 184, 0.11);
            background: rgba(148, 163, 184, 0.03);
        }

        .fs-def span {
            display: block;
            color: #d9e9fc;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .fs-def p {
            color: #8398b3;
            font-size: 0.73rem;
            line-height: 1.5;
            margin: 0 0 0.5rem;
        }

        .fs-def em {
            color: #8fd9e6;
            font-size: 0.68rem;
            font-style: normal;
            letter-spacing: 0.01em;
        }

        .fs-result {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.15rem 1.3rem;
            margin: 0.4rem 0 1rem;
            border-radius: 16px;
            animation: fs-rise 0.35s ease both;
        }

        .fs-result-danger {
            background: linear-gradient(115deg, rgba(251, 113, 133, 0.16), rgba(13, 25, 44, 0.92) 62%);
            border: 1px solid rgba(251, 113, 133, 0.36);
        }

        .fs-result-safe {
            background: linear-gradient(115deg, rgba(45, 212, 191, 0.14), rgba(13, 25, 44, 0.92) 62%);
            border: 1px solid rgba(45, 212, 191, 0.32);
        }

        .fs-signal {
            width: 34px;
            height: 34px;
            flex: 0 0 34px;
            display: grid;
            place-items: center;
            border-radius: 50%;
        }

        .fs-result-danger .fs-signal {
            border: 1px solid rgba(251, 113, 133, 0.46);
            background: rgba(251, 113, 133, 0.12);
        }

        .fs-result-safe .fs-signal {
            border: 1px solid rgba(45, 212, 191, 0.46);
            background: rgba(45, 212, 191, 0.11);
        }

        .fs-signal::after {
            content: "";
            width: 9px;
            height: 9px;
            border-radius: 50%;
        }

        .fs-result-danger .fs-signal::after {
            background: var(--fs-rose);
            box-shadow: 0 0 12px rgba(251, 113, 133, 0.75);
        }

        .fs-result-safe .fs-signal::after {
            background: var(--fs-teal);
            box-shadow: 0 0 12px rgba(45, 212, 191, 0.65);
        }

        .fs-result-title {
            color: #f6fbff;
            font-size: 1.12rem;
            font-weight: 720;
            letter-spacing: -0.02em;
        }

        .fs-result-sub {
            color: #93a8c2;
            font-size: 0.77rem;
            line-height: 1.5;
            margin-top: 0.2rem;
        }

        .fs-result-pct {
            margin-left: auto;
            color: #f3f9ff;
            font-size: 1.45rem;
            font-weight: 730;
            letter-spacing: -0.03em;
            white-space: nowrap;
        }

        .fs-prob-value {
            color: #f5faff;
            font-size: 2.35rem;
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.05;
            margin: 0.35rem 0 0.2rem;
        }

        .fs-gauge-track {
            position: relative;
            height: 11px;
            margin: 1.4rem 0 1.55rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.13);
        }

        .fs-gauge-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22d3ee, #2dd4bf);
            box-shadow: 0 0 16px rgba(34, 211, 238, 0.28);
        }

        .fs-gauge-fill.danger {
            background: linear-gradient(90deg, #f59e0b, #fb7185);
            box-shadow: 0 0 16px rgba(251, 113, 133, 0.25);
        }

        .fs-gauge-marker {
            position: absolute;
            top: -7px;
            width: 2px;
            height: 25px;
            background: #dbe8f8;
            opacity: 0.8;
        }

        .fs-gauge-marker span {
            position: absolute;
            top: 29px;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            color: #90a5bf;
            font-size: 0.63rem;
            font-weight: 600;
        }

        .fs-gauge-scale {
            display: flex;
            justify-content: space-between;
            color: #657b98;
            font-size: 0.61rem;
        }

        .fs-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
            margin-top: 0.9rem;
        }

        .fs-mini-grid > div {
            padding: 0.65rem 0.75rem;
            border-radius: 9px;
            background: rgba(148, 163, 184, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.09);
        }

        .fs-mini-grid span {
            display: block;
            color: #7287a3;
            font-size: 0.62rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .fs-mini-grid b {
            display: block;
            color: #dcebfb;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 0.22rem;
        }

        .fs-mini-grid b.danger {
            color: #fda0ad;
        }

        .fs-mini-grid b.safe {
            color: #8ce9dd;
        }

        .fs-facts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.9rem;
        }

        .fs-fact {
            flex: 1 1 120px;
            padding: 0.65rem 0.8rem;
            border-radius: 10px;
            background: rgba(148, 163, 184, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.10);
        }

        .fs-fact span {
            display: block;
            color: #7287a3;
            font-size: 0.61rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .fs-fact b {
            display: block;
            color: #dcebfb;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 0.2rem;
        }

        .fs-truth-value {
            color: #f5faff;
            font-size: 1.5rem;
            font-weight: 730;
            letter-spacing: -0.03em;
            margin: 0.2rem 0 0.45rem;
        }

        .fs-truth-value.danger {
            color: #fda0ad;
        }

        .fs-truth-value.safe {
            color: #8ce9dd;
        }

        .fs-agree {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.85rem;
            padding: 0.34rem 0.6rem;
            border-radius: 999px;
            font-size: 0.67rem;
            font-weight: 650;
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #a3b6ce;
            background: rgba(148, 163, 184, 0.05);
        }

        .fs-agree.ok {
            color: #9de9df;
            border-color: rgba(45, 212, 191, 0.22);
            background: rgba(45, 212, 191, 0.06);
        }

        .fs-agree.diff {
            color: #f8cf72;
            border-color: rgba(251, 191, 36, 0.22);
            background: rgba(251, 191, 36, 0.06);
        }

        .fs-warning {
            padding: 0.95rem 1.05rem;
            margin: 0.8rem 0;
            border-radius: 12px;
            border: 1px solid rgba(251, 191, 36, 0.20);
            background: rgba(251, 191, 36, 0.055);
            color: #c3ad83;
            font-size: 0.79rem;
            line-height: 1.6;
        }

        .fs-warning b {
            color: #f2d79d;
        }

        .fs-error {
            padding: 0.95rem 1.05rem;
            margin: 0.8rem 0;
            border-radius: 12px;
            border: 1px solid rgba(251, 113, 133, 0.25);
            background: rgba(251, 113, 133, 0.075);
            color: #e0a5ae;
            font-size: 0.8rem;
            line-height: 1.55;
        }

        .fs-class-row {
            display: grid;
            grid-template-columns: 105px minmax(80px, 1fr) 92px;
            gap: 0.75rem;
            align-items: center;
            margin: 0.7rem 0;
        }

        .fs-class-row label {
            color: #9db0c9;
            font-size: 0.74rem;
        }

        .fs-class-row .track {
            height: 12px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.10);
            overflow: hidden;
        }

        .fs-class-row .fill {
            height: 100%;
            border-radius: 999px;
        }

        .fs-class-row .fill.legit {
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
        }

        .fs-class-row .fill.fraud {
            background: linear-gradient(90deg, #f59e0b, #fb7185);
        }

        .fs-class-row b {
            color: #dcebfb;
            font-size: 0.76rem;
            text-align: right;
            font-weight: 650;
        }

        .fs-check-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.48rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            color: #8398b3;
            font-size: 0.75rem;
        }

        .fs-check-row:last-child {
            border-bottom: none;
        }

        .fs-check-row b {
            color: #dcebfb;
            font-weight: 650;
        }

        .fs-feature-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .fs-feature-chip {
            padding: 0.26rem 0.48rem;
            border-radius: 6px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(148, 163, 184, 0.04);
            color: #a8bdd6;
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            font-size: 0.67rem;
        }

        .fs-runbook {
            padding: 0.9rem 1rem;
            border-radius: 10px;
            border: 1px solid rgba(34, 211, 238, 0.14);
            background: rgba(34, 211, 238, 0.035);
            color: #9fc9d6;
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            font-size: 0.74rem;
            line-height: 1.8;
        }

        .fs-footer {
            margin-top: 3rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(148, 163, 184, 0.11);
            color: #637a97;
            font-size: 0.7rem;
            line-height: 1.6;
        }

        .fs-footer b {
            color: #9bb0ca;
        }

        .fs-dataset-source {
            margin-top: 1rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(148, 163, 184, 0.09);
            max-width: 72rem;
            font-size: 0.72rem;
        }

        .fs-dataset-source a {
            color: #22d3ee;
            text-decoration: none;
        }

        .fs-dataset-source a:hover {
            text-decoration: underline;
        }

        .fs-dataset-source code {
            color: #b9cbe3;
        }

        @keyframes fs-rise {
            from {
                opacity: 0;
                transform: translateY(5px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fs-side-in {
            from {
                opacity: 0;
                transform: translateX(-9px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fs-nav-shift {
            0%, 100% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
        }

        @keyframes fs-nav-glow {
            0%, 100% {
                box-shadow: 0 0 0 rgba(34, 211, 238, 0);
            }
            50% {
                box-shadow: 0 0 20px rgba(34, 211, 238, 0.10);
            }
        }

        @keyframes fs-nav-bar {
            0%, 100% {
                opacity: 0.65;
                transform: scaleY(0.78);
            }
            50% {
                opacity: 1;
                transform: scaleY(1);
            }
        }

        @keyframes fs-mark-float {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-2px);
            }
        }

        @keyframes fs-status-pulse {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.0);
            }
            50% {
                box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.08);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .fs-kpi, .fs-panel, .fs-result, .fs-brand, .fs-mark,
            .fs-nav-label, .fs-side-meta, .fs-dot-ok,
            [data-testid="stSidebar"] .stButton,
            [data-testid="stSidebar"] .stButton > button,
            [data-testid="stSidebar"] .stButton > button[kind="primary"],
            [data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
                animation: none;
                transition: none;
            }
        }

        @media (max-width: 760px) {
            .fs-def-grid {
                grid-template-columns: 1fr;
            }
            .fs-result-pct {
                display: none;
            }
            .fs-strip {
                grid-template-columns: 1fr;
            }
            .fs-class-row {
                grid-template-columns: 88px minmax(60px, 1fr) 78px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_page(page_name: str) -> None:
    """Callback: update navigation before Streamlit reruns the script."""
    st.session_state.page = page_name


def render_sidebar(artifact: dict | None) -> None:
    """Render branding, navigation, and real (not simulated) status details."""
    with st.sidebar:
        html_block(
            """
            <div class="fs-brand">
                <div class="fs-mark" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none">
                        <path d="M12 2.6 4.5 5.5v5.7c0 4.7 3.1 8.8 7.5 10.2 4.4-1.4 7.5-5.5 7.5-10.2V5.5L12 2.6Z"
                              stroke="currentColor" stroke-width="1.5"/>
                        <path d="m8.5 11.8 2.4 2.4 4.8-5.1" stroke="currentColor"
                              stroke-width="1.7" stroke-linecap="round"
                              stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <div class="fs-brand-name">FraudShield <span>AI</span></div>
                    <div class="fs-brand-tag">Intelligent Transaction Risk Analysis</div>
                </div>
            </div>
            """,
        )

        if artifact is not None:
            html_block('<div class="fs-nav-label">Workspace</div>')
            for page_name in NAV_ITEMS:
                is_active = st.session_state.get("page") == page_name
                st.button(
                    page_name,
                    key=f"nav_{page_name}",
                    type="primary" if is_active else "secondary",
                    width="stretch",
                    on_click=set_page,
                    args=(page_name,),
                )

        model_status = "Artifact loaded" if artifact is not None else "Artifact missing"
        model_dot = "fs-dot-ok" if artifact is not None else "fs-dot-bad"
        dataset_source = resolve_dataset()
        dataset_status = {
            "full": "Complete",
            "sample": "Sample preview",
        }.get(dataset_source["kind"], "Not found")
        threshold_text = (
            f"{artifact['threshold']:.2f}" if artifact is not None else "Unavailable"
        )
        feature_text = (
            str(len(artifact["feature_names"])) if artifact is not None else "Unavailable"
        )

        html_block(
            f"""
            <div class="fs-side-meta">
                <div class="fs-side-row"><span>Model</span><b>Random Forest</b></div>
                <div class="fs-side-row"><span>Threshold</span><b>{threshold_text}</b></div>
                <div class="fs-side-row"><span>Features</span><b>{feature_text}</b></div>
                <div class="fs-side-row"><span>Dataset CSV</span><b>{dataset_status}</b></div>
                <div class="fs-side-status">
                    <span class="fs-dot {model_dot}"></span>{model_status}
                </div>
                <div class="fs-side-note">
                    Educational ML prototype. Not a production banking security system.
                    No external network calls or live banking connections.
                </div>
            </div>
            """,
        )


def scroll_main_to_top() -> None:
    """Smoothly reset the main scroll container after sidebar navigation.

    Streamlit preserves the scroll position across reruns, which can make a
    sidebar click look like it did nothing when the user is already scrolled
    down. This runs only when the selected page actually changes.
    """
    st.html(
        """
        <script>
        (() => {
            const scrollToTop = () => {
                const root = window.parent.document;
                const main = root.querySelector('[data-testid="stMain"]')
                    || root.querySelector('.stMain');
                if (main) {
                    main.scrollTo({ top: 0, behavior: 'smooth' });
                }
            };
            requestAnimationFrame(() => {
                requestAnimationFrame(() => setTimeout(scrollToTop, 30));
            });
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    html_block(
        f"""
        <div class="fs-eyebrow">{esc(eyebrow)}</div>
        <div class="fs-page-title">{esc(title)}</div>
        <p class="fs-lead">{esc(subtitle)}</p>
        """,
    )


def render_notice(text: str) -> None:
    html_block(
        f"""
        <div class="fs-notice">
            <span class="fs-notice-mark"></span>
            <div>{text}</div>
        </div>
        """,
    )


def render_section_head(title: str, subtitle: str = "", badge: str = "") -> None:
    badge_html = (
        f'<span class="fs-badge"><span class="fs-badge-dot"></span>{esc(badge)}</span>'
        if badge
        else ""
    )
    html_block(
        f"""
        <div class="fs-section-head">
            <div>
                <div class="fs-section-title">{esc(title)}</div>
                <div class="fs-section-sub">{esc(subtitle)}</div>
            </div>
            {badge_html}
        </div>
        """,
    )


def render_kpi_row(items: list[dict]) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            html_block(
                f"""
                <div class="fs-kpi {esc(item.get('tone', ''))}">
                    <div class="fs-kpi-label">{esc(item['label'])}</div>
                    <div class="fs-kpi-value">{item['value']}</div>
                    <div class="fs-kpi-foot">{esc(item.get('foot', ''))}</div>
                </div>
                """,
            )


def confusion_matrix_html(tn: int, fp: int, fn: int, tp: int) -> str:
    """Build a readable confusion matrix with explicit cell meanings."""
    return f"""
    <div class="fs-cm" role="table" aria-label="Confusion matrix">
        <div class="fs-cm-axis fs-cm-axis-y">Actual</div>
        <div class="fs-cm-axis">Predicted<br/>legitimate</div>
        <div class="fs-cm-axis">Predicted<br/>fraud</div>

        <div class="fs-cm-axis fs-cm-axis-y">Legitimate</div>
        <div class="fs-cm-cell good">
            <b>{tn:,}</b><span>True negatives<br/>correctly cleared</span>
        </div>
        <div class="fs-cm-cell warn">
            <b>{fp:,}</b><span>False positives<br/>false alarms</span>
        </div>

        <div class="fs-cm-axis fs-cm-axis-y">Fraud</div>
        <div class="fs-cm-cell bad">
            <b>{fn:,}</b><span>False negatives<br/>missed fraud</span>
        </div>
        <div class="fs-cm-cell blue">
            <b>{tp:,}</b><span>True positives<br/>fraud caught</span>
        </div>
    </div>
    """


def gauge_html(probability: float, threshold: float) -> str:
    fill_pct = max(0.0, min(100.0, probability * 100.0))
    threshold_pct = max(0.0, min(100.0, threshold * 100.0))
    tone = "danger" if probability >= threshold else ""
    return f"""
    <div class="fs-gauge-track" role="img"
         aria-label="Fraud probability {fmt_probability(probability)} against threshold {threshold:.2%}">
        <div class="fs-gauge-fill {tone}" style="width:{fill_pct:.3f}%"></div>
        <div class="fs-gauge-marker" style="left:{threshold_pct:.3f}%">
            <span>Threshold {threshold:.2%}</span>
        </div>
    </div>
    <div class="fs-gauge-scale"><span>0%</span><span>100%</span></div>
    """


def render_result_banner(probability: float, prediction: int, threshold: float) -> None:
    if prediction == 1:
        css_class = "fs-result-danger"
        title = "Potential fraud"
        subtitle = (
            f"The estimated probability is at or above the {threshold:.2%} "
            "decision threshold."
        )
    else:
        css_class = "fs-result-safe"
        title = "Predicted legitimate"
        subtitle = (
            f"The estimated probability is below the {threshold:.2%} "
            "decision threshold."
        )

    html_block(
        f"""
        <div class="fs-result {css_class}">
            <span class="fs-signal" aria-hidden="true"></span>
            <div>
                <div class="fs-result-title">{title}</div>
                <div class="fs-result-sub">{esc(subtitle)}</div>
            </div>
            <div class="fs-result-pct">{fmt_probability(probability)}</div>
        </div>
        """,
    )


def render_probability_panel(probability: float, prediction: int, threshold: float) -> None:
    decision_text = "Potential fraud" if prediction == 1 else "Predicted legitimate"
    decision_class = "danger" if prediction == 1 else "safe"
    html_block(
        f"""
        <div class="fs-panel">
            <div class="fs-panel-title">Model estimate</div>
            <div class="fs-prob-value">{fmt_probability(probability)}</div>
            <div class="fs-muted">Estimated probability of fraud</div>
            {gauge_html(probability, threshold)}
            <div class="fs-mini-grid">
                <div><span>Decision threshold</span><b>{threshold:.2%}</b></div>
                <div><span>Classified as</span><b class="{decision_class}">{decision_text}</b></div>
            </div>
        </div>
        """,
    )


def render_truth_panel(actual_label: int, prediction: int) -> None:
    is_fraud = actual_label == 1
    truth_text = "Fraud" if is_fraud else "Legitimate"
    truth_class = "danger" if is_fraud else "safe"
    agrees = int(actual_label) == int(prediction)
    agreement_class = "ok" if agrees else "diff"
    agreement_text = (
        "Model decision matches the dataset label"
        if agrees
        else "Model decision differs from the dataset label"
    )

    html_block(
        f"""
        <div class="fs-panel">
            <div class="fs-panel-title">Dataset label</div>
            <div class="fs-truth-value {truth_class}">{truth_text}</div>
            <div class="fs-muted">
                Ground truth from the <code>Class</code> column. It is shown for
                comparison only and is never passed to the model.
            </div>
            <span class="fs-agree {agreement_class}">{esc(agreement_text)}</span>
        </div>
        """,
    )


def render_input_panel(values: dict) -> None:
    time_value = float(values.get("Time", 0.0))
    amount_value = float(values.get("Amount", 0.0))
    html_block(
        f"""
        <div class="fs-panel">
            <div class="fs-panel-title">Input contract</div>
            <div class="fs-check-row"><span>Features supplied</span><b>30 / 30</b></div>
            <div class="fs-check-row"><span>Column order</span><b>Matches artifact</b></div>
            <div class="fs-check-row"><span>Numeric validation</span><b>Passed</b></div>
            <div class="fs-check-row"><span>Time value</span><b>{time_value:,.0f} s</b></div>
            <div class="fs-check-row"><span>Amount value</span><b>{amount_value:,.2f}</b></div>
            <div class="fs-muted" style="margin-top:0.7rem;">
                No ground-truth label is available for a custom vector.
            </div>
        </div>
        """,
    )


def render_transaction_facts(row: pd.Series, row_index: int) -> None:
    actual_label = int(row[TARGET_COLUMN])
    actual_text = "Fraud" if actual_label == 1 else "Legitimate"
    actual_class = "danger" if actual_label == 1 else "safe"
    html_block(
        f"""
        <div class="fs-facts">
            <div class="fs-fact"><span>Dataset row</span><b>#{int(row_index):,}</b></div>
            <div class="fs-fact"><span>Time value</span><b>{float(row['Time']):,.0f} s</b></div>
            <div class="fs-fact"><span>Amount</span><b>{float(row['Amount']):,.2f}</b></div>
            <div class="fs-fact"><span>Dataset label</span>
                <b class="{actual_class}" style="color: {'#fda0ad' if actual_label == 1 else '#8ce9dd'};">
                    {actual_text}
                </b>
            </div>
        </div>
        """,
    )


def render_probability_disclaimer() -> None:
    st.caption(
        "This probability is a model estimate, not a guarantee. The decision uses the "
        "saved threshold and should be reviewed by a human in any real risk process."
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def render_overview(artifact: dict) -> None:
    html_block(
        f"""
        <div class="fs-hero">
            <div class="fs-eyebrow">Intelligent Transaction Risk Analysis</div>
            <h1>Fraud risk,<br/><span class="fs-gradient">clearly quantified.</span></h1>
            <p>
                FraudShield AI turns the saved Random Forest model into a focused analytics
                workspace: inspect verified test performance, score individual transactions,
                and understand the data behind the model.
            </p>
            <div class="fs-chip-row">
                <span class="fs-chip">Random Forest</span>
                <span class="fs-chip">Artifact threshold {artifact_threshold_text(artifact)}</span>
                <span class="fs-chip">{len(artifact['feature_names'])} input features</span>
                <span class="fs-chip">Held-out test results</span>
            </div>
        </div>
        """,
    )
    render_notice(
        "<b>Educational ML prototype.</b> All metrics on this page come from a held-out "
        "test evaluation of the saved model. This dashboard has no live transactions, "
        "banking integrations, or production security claims."
    )
    render_threshold_mismatch_warning(artifact)

    render_section_head(
        "Verified test performance",
        f"Held-out test split, evaluation threshold {evaluation_threshold_text()}",
        "Evaluated",
    )
    render_kpi_row(
        [
            {
                "label": "Precision",
                "value": f"{TEST_RESULTS['precision']:.2%}",
                "foot": "57 correct fraud alerts out of 69 alerts raised.",
            },
            {
                "label": "Recall",
                "value": f"{TEST_RESULTS['recall']:.2%}",
                "foot": "57 of 71 actual fraud cases were detected.",
                "tone": "teal",
            },
            {
                "label": "F1-score",
                "value": f"{TEST_RESULTS['f1']:.4f}",
                "foot": "Balance between precision and recall.",
                "tone": "blue",
            },
            {
                "label": "Average precision",
                "value": f"{TEST_RESULTS['average_precision']:.4f}",
                "foot": "Ranking quality across precision-recall thresholds.",
                "tone": "amber",
            },
        ],
    )

    left, right = st.columns([1.18, 1])
    with left:
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Test confusion matrix</div>
                {confusion_matrix_html(
                    TEST_RESULTS['tn'],
                    TEST_RESULTS['fp'],
                    TEST_RESULTS['fn'],
                    TEST_RESULTS['tp'],
                )}
                <div class="fs-muted" style="margin-top:0.85rem;">
                    42,559 held-out test transactions. The model raised 69 alerts in total:
                    57 were genuine fraud and 12 were false alarms.
                </div>
            </div>
            """,
        )

    with right:
        fraud_rate = TEST_RESULTS["actual_fraud"] / TEST_RESULTS["transactions"]
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Evaluation context</div>
                <div class="fs-strip" style="margin-top:0;">
                    <div class="fs-strip-item"><span>Test transactions</span><b>42,559</b></div>
                    <div class="fs-strip-item"><span>Actual fraud</span><b>71</b></div>
                    <div class="fs-strip-item"><span>Fraud rate</span><b>{fraud_rate:.2%}</b></div>
                </div>
                <div class="fs-mini-grid" style="margin-top:0.75rem;">
                    <div><span>Model</span><b>Random Forest</b></div>
                    <div><span>Trees</span><b>100</b></div>
                    <div><span>Class weighting</span><b>Balanced</b></div>
                    <div><span>Threshold source</span><b>Validation split</b></div>
                </div>
                <div class="fs-muted" style="margin-top:0.8rem;">
                    The threshold was selected on the validation split and then applied
                    unchanged to the held-out test set. It was not tuned on test results.
                </div>
            </div>
            """,
        )

    render_section_head("How to read the metrics", "Plain-language definitions")
    definitions = "".join(
        f"""
        <div class="fs-def">
            <span>{esc(name)}</span>
            <p>{esc(description)}</p>
            <em>{esc(detail)}</em>
        </div>
        """
        for name, description, detail in METRIC_DEFINITIONS
    )
    html_block(f'<div class="fs-panel"><div class="fs-def-grid">{definitions}</div></div>')

    render_notice(
        "The test set contains only 71 fraud cases, so a single case moves recall by about "
        "1.4 percentage points. Treat small metric differences as uncertain rather than "
        "as proof of a large performance difference."
    )


def set_random_sample(pool: str) -> None:
    """Callback: choose a real row from the active dataset."""
    source = resolve_dataset()
    if source["kind"] == "missing":
        return

    data, _ = load_dataset(str(source["path"]))
    if pool == "fraud":
        candidates = data[data[TARGET_COLUMN] == 1]
    elif pool == "legitimate":
        candidates = data[data[TARGET_COLUMN] == 0]
    else:
        candidates = data

    if len(candidates) > 0:
        st.session_state.sample_index = int(candidates.sample(n=1).index[0])


def render_sample_mode(artifact: dict) -> None:
    features = artifact["feature_names"]
    threshold = artifact["threshold"]
    source = resolve_dataset()

    if source["kind"] == "missing":
        html_block(
            """
            <div class="fs-warning">
                <b>creditcard.csv was not found, and the bundled preview
                data/sample_transactions.csv is also unavailable.</b>
                Sample-transaction mode is unavailable. Custom feature input remains
                usable, and the Overview and Model Performance sections do not require
                the dataset.
            </div>
            """,
        )
        return

    try:
        data, summary = load_dataset(str(source["path"]))
    except Exception as error:  # noqa: BLE001 - surface a clear UI error
        st.error(f"The dataset could not be loaded: {error}")
        return

    if source["is_sample"]:
        render_notice(
            "<b>Limited preview.</b> The complete <code>creditcard.csv</code> is not "
            "available here, so sample mode uses the bundled "
            "<code>data/sample_transactions.csv</code> file. It contains a small "
            f"selection of {summary['rows']:,} rows, not the full dataset. Predictions "
            "and dataset labels below describe only this preview."
        )

    missing_features = [name for name in features if name not in data.columns]
    if missing_features:
        st.error(
            "The dataset is missing model features: "
            + ", ".join(missing_features)
            + "."
        )
        return
    if TARGET_COLUMN not in data.columns:
        st.error("The dataset is missing the required Class column.")
        return

    if summary["rows"] == 0:
        st.error(
            f"{source['label']} contains no rows. "
            "Sample-transaction mode is unavailable."
        )
        st.info(
            "Custom feature input remains usable, and the Overview and Model "
            "Performance sections do not require dataset rows."
        )
        return

    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0
    # Keep a stale index from a previously loaded file within the active range.
    if not 0 <= int(st.session_state.sample_index) < len(data):
        st.session_state.sample_index = 0

    if source["is_sample"]:
        subtitle = (
            f"{summary['rows']:,} bundled preview rows available (not the complete "
            "dataset). Row selection is local and never transmitted."
        )
    else:
        subtitle = (
            f"{summary['rows']:,} local rows available. Row selection is local and "
            "never transmitted."
        )

    render_section_head("Choose a transaction", subtitle, source["badge"])

    control_left, control_mid, control_right = st.columns([2.25, 1, 1])
    with control_left:
        selected_index = st.number_input(
            "Dataset row index",
            min_value=0,
            max_value=int(len(data) - 1),
            step=1,
            key="sample_index",
            help="Select any row by its loaded dataset index. The row's Class label is not used by the model.",
        )
    with control_mid:
        st.button(
            "Random row",
            width="stretch",
            on_click=set_random_sample,
            args=("any",),
            help="Pick a random row from the active dataset.",
        )
    with control_right:
        st.button(
            "Random fraud",
            width="stretch",
            on_click=set_random_sample,
            args=("fraud",),
            help="Pick a random row whose dataset label is fraud.",
        )

    try:
        row = data.loc[int(selected_index)]
    except KeyError:
        st.error("That row index is not present in the loaded dataset.")
        return

    values = model_input_values(row, features)
    try:
        probability, prediction = predict_transaction(
            artifact["model"], features, threshold, values
        )
    except Exception as error:  # noqa: BLE001 - show a safe inference error
        st.error(f"The transaction could not be scored: {error}")
        return

    render_transaction_facts(row, int(selected_index))
    render_result_banner(probability, prediction, threshold)

    left, right = st.columns([1.05, 1])
    with left:
        render_probability_panel(probability, prediction, threshold)
    with right:
        render_truth_panel(int(row[TARGET_COLUMN]), prediction)

    render_probability_disclaimer()

    with st.expander("View all 30 model inputs for this transaction"):
        formula_table = pd.DataFrame(
            {
                "Model order": range(1, len(features) + 1),
                "Feature": features,
                "Value": [float(row[name]) for name in features],
            }
        )
        st.dataframe(
            formula_table,
            width="stretch",
            hide_index=True,
            height=360,
            column_config={
                "Model order": st.column_config.NumberColumn(format="%d"),
                "Value": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.caption(
            "V1-V28 are anonymized PCA components from the original dataset. Time is "
            "seconds elapsed since the first record; Amount is in the dataset's raw units."
        )


def clear_custom_state() -> None:
    """Callback: clear the advanced form before widgets are re-instantiated.

    Widget values inside an ``st.form`` are retained by the form, so deleting
    their session-state keys alone does not visually reset them. Bumping a
    version used in the form and widget keys forces Streamlit to create fresh
    widgets with their default (empty) values on the next rerun.
    """
    st.session_state.custom_form_version = (
        int(st.session_state.get("custom_form_version", 0)) + 1
    )
    for key in list(st.session_state.keys()):
        if key.startswith("custom_feature_"):
            del st.session_state[key]
    st.session_state.pop("custom_result", None)
    st.session_state.pop("custom_error", None)
    st.session_state.pop("custom_missing", None)


def render_custom_mode(artifact: dict) -> None:
    features = artifact["feature_names"]
    threshold = artifact["threshold"]
    form_version = int(st.session_state.get("custom_form_version", 0))

    html_block(
        """
        <div class="fs-warning">
            <b>Advanced input.</b> The model requires all 30 numeric features in the
            artifact's exact order. V1-V28 are anonymized PCA components, not raw
            business fields, so manual entry is only useful when you already have a
            prepared feature vector. No default transaction values are pre-filled.
        </div>
        """,
    )

    reset_col, _ = st.columns([1, 3])
    with reset_col:
        st.button(
            "Reset fields",
            width="stretch",
            on_click=clear_custom_state,
        )

    values: dict[str, float | None] = {}
    with st.form(f"custom_feature_form_{form_version}"):
        st.markdown("#### Feature vector")
        st.caption(
            "Enter every value in the model's feature list. Empty fields are rejected."
        )
        columns = st.columns(3)
        for index, name in enumerate(features):
            label = name
            if name == "Time":
                label = "Time (seconds)"
            elif name == "Amount":
                label = "Amount (raw units)"
            with columns[index % 3]:
                values[name] = st.number_input(
                    label,
                    value=None,
                    step=0.1,
                    format="%.4f",
                    key=f"custom_feature_{form_version}_{name}",
                    help="PCA component. Enter the transformed numeric value." if name.startswith("V") else None,
                )

        submitted = st.form_submit_button(
            "Analyze transaction",
            type="primary",
            width="stretch",
        )

    if submitted:
        missing = [name for name, value in values.items() if value is None]
        st.session_state.pop("custom_result", None)
        st.session_state.pop("custom_error", None)
        if missing:
            st.session_state.custom_missing = missing
            st.session_state.custom_error = (
                f"{len(missing)} of 30 required features are empty."
            )
        else:
            try:
                probability, prediction = predict_transaction(
                    artifact["model"],
                    features,
                    threshold,
                    {name: float(value) for name, value in values.items()},
                )
            except Exception as error:  # noqa: BLE001 - show a safe inference error
                st.session_state.custom_error = str(error)
            else:
                st.session_state.custom_result = {
                    "probability": probability,
                    "prediction": prediction,
                    "values": {name: float(value) for name, value in values.items()},
                }
                st.session_state.pop("custom_missing", None)

    if "custom_error" in st.session_state:
        html_block(
            f'<div class="fs-error">{esc(st.session_state.custom_error)}</div>',
        )
        missing = st.session_state.get("custom_missing")
        if missing:
            with st.expander("Missing features"):
                st.write(", ".join(missing))

    result = st.session_state.get("custom_result")
    if result:
        render_section_head(
            "Result for the last submitted vector",
            "Custom input, exact model feature order",
        )
        render_result_banner(result["probability"], result["prediction"], threshold)
        left, right = st.columns([1.05, 1])
        with left:
            render_probability_panel(
                result["probability"], result["prediction"], threshold
            )
        with right:
            render_input_panel(result["values"])
        render_probability_disclaimer()

        with st.expander("Review the submitted feature vector"):
            submitted_table = pd.DataFrame(
                {
                    "Model order": range(1, len(features) + 1),
                    "Feature": features,
                    "Value": [result["values"][name] for name in features],
                }
            )
            st.dataframe(
                submitted_table,
                width="stretch",
                hide_index=True,
                height=360,
                column_config={
                    "Model order": st.column_config.NumberColumn(format="%d"),
                    "Value": st.column_config.NumberColumn(format="%.6f"),
                },
            )


def render_risk_analysis(artifact: dict) -> None:
    render_page_header(
        "Transaction Risk Analysis",
        "Score a single transaction",
        "Inspect a row from the loaded dataset (the complete CSV when present, otherwise "
        "the bundled sample preview), or enter a complete 30-feature vector. The saved "
        "threshold decides the final class.",
    )

    mode = st.radio(
        "Analysis mode",
        ["Sample transaction", "Custom feature input"],
        horizontal=True,
        label_visibility="collapsed",
        key="analysis_mode",
    )

    if mode == "Sample transaction":
        render_sample_mode(artifact)
    else:
        render_custom_mode(artifact)


def render_dataset_explorer() -> None:
    render_page_header(
        "Dataset Explorer",
        "Dataset currently loaded",
        "A lightweight summary of the data available to the dashboard. The Class column "
        "is summarized here and is never used as a model input.",
    )

    source = resolve_dataset()
    if source["kind"] == "missing":
        st.error(
            "creditcard.csv was not found in the project folder, and the bundled sample "
            "data/sample_transactions.csv is also unavailable. The dataset explorer is "
            "unavailable, but the Overview and Model Performance sections still work."
        )
        return

    try:
        data, summary = load_dataset(str(source["path"]))
    except Exception as error:  # noqa: BLE001 - surface a clear UI error
        st.error(f"The dataset could not be loaded: {error}")
        return

    if source["is_sample"]:
        render_notice(
            "<b>Limited preview, not the complete dataset.</b> The full "
            "<code>creditcard.csv</code> is not present, so this page reads the bundled "
            f"<code>{source['label']}</code> file ({summary['rows']:,} rows). Every count, "
            "chart, and fraud statistic below describes only this small sample. The "
            "sample includes both classes so fraud cases stay visible; it does not "
            "preserve the original dataset's class balance."
        )
    else:
        render_notice(
            "<b>Complete local dataset.</b> The dashboard is reading "
            "<code>creditcard.csv</code> from the project folder. All counts and "
            "statistics on this page are calculated from that file."
        )

    render_section_head(
        "Dataset at a glance",
        f"{source['label']} · Local file, not a live feed",
        source["badge"],
    )
    render_kpi_row(
        [
            {
                "label": "Rows",
                "value": f"{summary['rows']:,}",
                "foot": (
                    "Rows in the bundled preview only."
                    if source["is_sample"]
                    else f"{summary['deduplicated_rows']:,} rows remain after removing duplicates."
                ),
            },
            {
                "label": "Columns",
                "value": str(summary["columns"]),
                "foot": f"{summary['columns']} loaded columns; the model expects 30 features plus Class.",
                "tone": "blue",
            },
            {
                "label": "Fraud rate",
                "value": f"{summary['fraud_rate']:.2%}",
                "foot": (
                    f"{summary['fraud_count']:,} fraud rows in this preview (not the full-dataset rate)."
                    if source["is_sample"]
                    else f"{summary['fraud_count']:,} fraudulent transactions."
                ),
                "tone": "amber",
            },
            {
                "label": "Duplicate rows",
                "value": f"{summary['duplicate_rows']:,}",
                "foot": "Exact duplicates in the loaded CSV.",
                "tone": "teal",
            },
        ],
    )

    left, right = st.columns([1.25, 1])
    with left:
        max_count = max(summary["legitimate_count"], summary["fraud_count"], 1)
        legit_log_width = 100.0
        fraud_log_width = (
            math.log10(summary["fraud_count"] + 1)
            / math.log10(max_count + 1)
            * 100.0
        )
        if source["is_sample"]:
            distribution_note = (
                "Bars use a log scale so fraud rows remain visible. Counts are exact "
                f"for this bundled preview. Fraud makes up {summary['fraud_rate']:.2%} "
                "of these sample rows; that is not the original dataset's fraud rate."
            )
        else:
            distribution_note = (
                "Bars use a log scale so the rare fraud cases remain visible. Counts are "
                f"exact. Fraud makes up {summary['fraud_rate']:.2%} of all rows, a severe "
                "class imbalance."
            )
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Class distribution</div>
                <div class="fs-class-row">
                    <label>Legitimate</label>
                    <div class="track">
                        <div class="fill legit" style="width:{legit_log_width:.2f}%"></div>
                    </div>
                    <b>{summary['legitimate_count']:,}</b>
                </div>
                <div class="fs-class-row">
                    <label>Fraud</label>
                    <div class="track">
                        <div class="fill fraud" style="width:{fraud_log_width:.2f}%"></div>
                    </div>
                    <b>{summary['fraud_count']:,}</b>
                </div>
                <div class="fs-muted" style="margin-top:0.75rem;">
                    {esc(distribution_note)}
                </div>
            </div>
            """,
        )

    with right:
        missing_text = (
            "None" if summary["missing_total"] == 0 else f"{summary['missing_total']:,}"
        )
        non_target_columns = sum(1 for column in data.columns if column != TARGET_COLUMN)
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Data quality</div>
                <div class="fs-check-row"><span>Loaded file</span><b>{esc(source['label'])}</b></div>
                <div class="fs-check-row"><span>Missing values</span><b>{missing_text}</b></div>
                <div class="fs-check-row"><span>Exact duplicate rows</span><b>{summary['duplicate_rows']:,}</b></div>
                <div class="fs-check-row"><span>Rows after deduplication</span><b>{summary['deduplicated_rows']:,}</b></div>
                <div class="fs-check-row"><span>Columns other than Class</span><b>{non_target_columns}</b></div>
                <div class="fs-check-row"><span>Target</span><b>Class (0 / 1)</b></div>
                <div class="fs-check-row"><span>DataFrame memory (approx.)</span><b>{summary['memory_mb']:.1f} MB</b></div>
            </div>
            """,
        )

    with st.expander("Column schema and data types"):
        schema = pd.DataFrame(
            {
                "Column": data.columns,
                "Dtype": [str(dtype) for dtype in data.dtypes],
                "Role": [
                    "Target label - never an input"
                    if column == TARGET_COLUMN
                    else "Input feature"
                    for column in data.columns
                ],
                "Missing": [int(count) for count in data.isna().sum().to_numpy()],
            }
        )
        st.dataframe(schema, width="stretch", hide_index=True, height=430)

    with st.expander("Preview the first rows"):
        st.dataframe(data.head(8), width="stretch")
        if source["is_sample"]:
            st.caption(
                "Preview of the bundled sample only. The model input excludes the Class "
                "column and uses the feature order stored inside the saved artifact."
            )
        else:
            st.caption(
                "Preview only. The model input excludes the Class column and uses the "
                "feature order stored inside the saved artifact."
            )

    if summary["missing_total"] > 0:
        st.warning(
            "Missing values were found. The dashboard does not impute them; inference "
            "validation rejects incomplete numeric vectors."
        )


def render_model_performance(artifact: dict) -> None:
    render_page_header(
        "Model Performance",
        "Held-out evaluation",
        "Verified results from the final test evaluation at the historical "
        "evaluation threshold. The test split was not used to select the threshold.",
    )
    render_notice(
        "<b>Verified evaluation, not a live score.</b> These numbers are fixed results "
        "from the held-out test set. Running this dashboard does not retrain the model."
    )
    render_threshold_mismatch_warning(artifact)

    total = TEST_RESULTS["transactions"]
    accuracy = (TEST_RESULTS["tn"] + TEST_RESULTS["tp"]) / total
    legitimate_baseline = (
        TEST_RESULTS["tn"] + TEST_RESULTS["fp"]
    ) / total
    alert_count = TEST_RESULTS["tp"] + TEST_RESULTS["fp"]

    render_section_head(
        "Operating point",
        f"Evaluation threshold {evaluation_threshold_text()}, "
        "selected on the validation split",
        "Test verified",
    )
    render_kpi_row(
        [
            {
                "label": "Precision",
                "value": f"{TEST_RESULTS['precision']:.2%}",
                "foot": f"{TEST_RESULTS['tp']} true frauds among {alert_count} alerts.",
            },
            {
                "label": "Recall",
                "value": f"{TEST_RESULTS['recall']:.2%}",
                "foot": f"{TEST_RESULTS['fn']} missed frauds and {TEST_RESULTS['fp']} false alarms.",
                "tone": "teal",
            },
            {
                "label": "F1-score",
                "value": f"{TEST_RESULTS['f1']:.4f}",
                "foot": "Harmonic mean of precision and recall.",
                "tone": "blue",
            },
            {
                "label": "Average precision",
                "value": f"{TEST_RESULTS['average_precision']:.4f}",
                "foot": "Threshold-independent ranking summary.",
                "tone": "amber",
            },
        ],
    )

    left, right = st.columns([1.1, 1])
    with left:
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Confusion matrix</div>
                {confusion_matrix_html(
                    TEST_RESULTS['tn'],
                    TEST_RESULTS['fp'],
                    TEST_RESULTS['fn'],
                    TEST_RESULTS['tp'],
                )}
                <div class="fs-muted" style="margin-top:0.85rem;">
                    {TEST_RESULTS['tp']:,} frauds caught and {TEST_RESULTS['fn']:,} missed at
                    the {evaluation_threshold_text()} evaluation threshold, out of
                    {TEST_RESULTS['actual_fraud']} actual fraud cases.
                </div>
            </div>
            """,
        )

    with right:
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Error analysis</div>
                <div class="fs-def" style="margin-bottom:0.55rem;">
                    <span>False positives: {TEST_RESULTS['fp']}</span>
                    <p>Legitimate transactions flagged as fraud. These create customer
                    friction and manual-review work.</p>
                </div>
                <div class="fs-def">
                    <span>False negatives: {TEST_RESULTS['fn']}</span>
                    <p>Fraudulent transactions the model missed. In fraud detection these
                    are often the more expensive error.</p>
                </div>
                <div class="fs-muted" style="margin-top:0.75rem;">
                    False negatives are the main reason recall is {TEST_RESULTS['recall']:.2%}
                    rather than 100%.
                </div>
            </div>
            """,
        )

    render_section_head("Context and limits", "Why headline accuracy is not enough")
    col1, col2, col3 = st.columns(3)
    with col1:
        html_block(
            f"""
            <div class="fs-panel" style="min-height:190px;">
                <div class="fs-panel-title">Threshold policy</div>
                <p class="fs-muted">{evaluation_threshold_text()} was selected using
                validation-set F1, then applied unchanged to the held-out test set.</p>
                <div class="fs-muted" style="margin-top:0.6rem;">
                    A different threshold would change the balance between false alarms and
                    missed fraud. The dashboard does not tune the threshold on test data.
                </div>
            </div>
            """,
        )
    with col2:
        html_block(
            f"""
            <div class="fs-panel" style="min-height:190px;">
                <div class="fs-panel-title">Small fraud sample</div>
                <p class="fs-muted">Only {TEST_RESULTS['actual_fraud']} fraud cases are in the
                test set. One additional miss changes recall by roughly
                {100 / TEST_RESULTS['actual_fraud']:.1f} percentage points.</p>
                <div class="fs-muted" style="margin-top:0.6rem;">
                    Treat small metric differences as uncertainty, not certainty.
                </div>
            </div>
            """,
        )
    with col3:
        html_block(
            f"""
            <div class="fs-panel" style="min-height:190px;">
                <div class="fs-panel-title">Accuracy caveat</div>
                <p class="fs-muted">Overall accuracy is {accuracy:.2%}, but a model that
                predicts every transaction as legitimate would already reach
                {legitimate_baseline:.2%}.</p>
                <div class="fs-muted" style="margin-top:0.6rem;">
                    Accuracy is not a headline metric here because it hides the
                    {TEST_RESULTS['fn']} missed frauds.
                </div>
            </div>
            """,
        )


def render_model_details(artifact: dict) -> None:
    features = artifact["feature_names"]
    threshold = artifact["threshold"]

    render_page_header(
        "Model Details",
        "Artifact and input contract",
        "Everything the dashboard knows about the saved model, including how to "
        "reproduce the evaluation and what this prototype does not do.",
    )
    render_threshold_mismatch_warning(artifact)

    left, right = st.columns(2)
    with left:
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Saved artifact</div>
                <div class="fs-check-row"><span>Algorithm</span><b>Random Forest classifier</b></div>
                <div class="fs-check-row"><span>Trees</span><b>100</b></div>
                <div class="fs-check-row"><span>Class weighting</span><b>Balanced</b></div>
                <div class="fs-check-row"><span>Live artifact threshold</span><b>{artifact_threshold_text(artifact)}</b></div>
                <div class="fs-check-row"><span>Feature count</span><b>{len(features)}</b></div>
                <div class="fs-check-row"><span>Artifact file</span><b>models/fraudshield_random_forest.joblib</b></div>
                <div class="fs-muted" style="margin-top:0.75rem;">
                    Loaded with <code>joblib.load()</code>. The app never calls
                    <code>fit()</code> and never writes over the artifact.
                </div>
            </div>
            """,
        )

    with right:
        html_block(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-title">Evaluation setup</div>
                <div class="fs-check-row"><span>Dataset</span><b>creditcard.csv</b></div>
                <div class="fs-check-row"><span>Preprocessing</span><b>Duplicate rows removed</b></div>
                <div class="fs-check-row"><span>Split</span><b>70% train / 15% validation / 15% test</b></div>
                <div class="fs-check-row"><span>Random state</span><b>42</b></div>
                <div class="fs-check-row"><span>Evaluation threshold</span><b>{evaluation_threshold_text()}</b></div>
                <div class="fs-check-row"><span>Test transactions</span><b>{TEST_RESULTS['transactions']:,}</b></div>
                <div class="fs-check-row"><span>Actual test frauds</span><b>{TEST_RESULTS['actual_fraud']}</b></div>
                <div class="fs-muted" style="margin-top:0.75rem;">
                    Held-out test metrics are displayed as verified constants from
                    <code>07_final_evaluation.py</code>. They remain tied to the
                    {evaluation_threshold_text()} evaluation threshold.
                </div>
            </div>
            """,
        )

    render_section_head(
        "Input features",
        "Order is taken directly from the saved artifact",
    )
    chips = "".join(
        f'<span class="fs-feature-chip">{esc(name)}</span>' for name in features
    )
    html_block(f'<div class="fs-panel"><div class="fs-feature-chips">{chips}</div></div>')

    render_section_head("Runbook", "Commands are run from the project folder")
    html_block(
        """
        <div class="fs-panel">
            <div class="fs-runbook">
                pip install -r requirements.txt<br/>
                python -m streamlit run 10_app.py
            </div>
            <div class="fs-muted" style="margin-top:0.7rem;">
                If the model artifact is missing, recreate it with
                <code>python 08_save_model.py</code>. The dashboard intentionally does not
                train a replacement model. <code>python 09_test_saved_model.py</code> verifies
                the saved file before you launch the app.
            </div>
        </div>
        """,
    )

    render_section_head("Limitations and responsible use")
    html_block(
        """
        <div class="fs-panel">
            <div class="fs-def-grid">
                <div class="fs-def">
                    <span>No live data</span>
                    <p>All analysis uses the local creditcard.csv file when present, or the
                    bundled sample preview otherwise. There is no live transaction feed,
                    bank connection, or external API call.</p>
                </div>
                <div class="fs-def">
                    <span>Dataset-specific</span>
                    <p>V1-V28 are anonymized PCA components from one historical dataset.
                    The model may not transfer to another bank, region, or time period.</p>
                </div>
                <div class="fs-def">
                    <span>Probability is not certainty</span>
                    <p>A model score is an estimate. A 0.19 result is still a legitimate
                    decision under the current policy, not evidence of safety.</p>
                </div>
                <div class="fs-def">
                    <span>Human review required</span>
                    <p>This prototype must not be used as the sole basis for blocking
                    transactions, denying service, or making financial decisions.</p>
                </div>
            </div>
        </div>
        """,
    )


def render_missing_model(error_type: str, detail: str = "") -> None:
    render_page_header(
        "Model Artifact",
        "Saved model required",
        "The dashboard is inference-only, so it cannot continue without the existing "
        "Random Forest artifact.",
    )
    st.error(
        "The saved model could not be loaded. The dashboard will not train a replacement "
        "model automatically."
    )
    html_block(
        """
        <div class="fs-panel">
            <div class="fs-panel-title">Expected artifact</div>
            <div class="fs-runbook">models/fraudshield_random_forest.joblib</div>
            <div class="fs-muted" style="margin-top:0.8rem;">
                Run the training/saving script once from the project folder, then reload
                this page.
            </div>
            <div class="fs-runbook" style="margin-top:0.7rem;">python 08_save_model.py</div>
            <div class="fs-muted" style="margin-top:0.8rem;">
                You can verify the saved file with
                <code>python 09_test_saved_model.py</code>.
            </div>
        </div>
        """,
    )
    if error_type != "missing":
        with st.expander("Technical detail"):
            st.code(detail or "Unknown load error.", language=None)


def render_footer() -> None:
    html_block(
        """
        <div class="fs-footer">
            <b>FraudShield AI</b> · Intelligent Transaction Risk Analysis<br/>
            Educational ML prototype. No live transactions, no external network calls,
            and no production banking security claims. Metrics describe a held-out test
            evaluation of the saved model.
            <div class="fs-dataset-source" id="dataset-source">
                <b>Dataset Source</b><br/>
                FraudShield AI uses the
                <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
                   target="_blank" rel="noopener noreferrer">Credit Card Fraud Detection</a>
                dataset published on Kaggle by the Machine Learning Group (ULB), in
                collaboration with Worldline. The original dataset contains 284,807
                transactions and 31 columns, including the target column
                <code>Class</code>: <code>Class = 1</code> means fraud and
                <code>Class = 0</code> means a legitimate transaction. It includes
                anonymized features <code>V1</code>-<code>V28</code>, plus
                <code>Time</code> and <code>Amount</code>. This project removes exact
                duplicate rows before training, so the processed row count can differ
                from the original dataset size. The dashboard reads the local
                <code>creditcard.csv</code> when it is available and otherwise falls
                back to the small committed preview at
                <code>data/sample_transactions.csv</code>. It does not download data from Kaggle.
            </div>
        </div>
        """,
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="FraudShield AI",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_css()

    if "page" not in st.session_state or st.session_state.page not in NAV_ITEMS:
        st.session_state.page = NAV_ITEMS[0]
    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0

    artifact = None
    load_error: Exception | None = None
    try:
        artifact = load_model_artifact(str(MODEL_PATH))
    except FileNotFoundError as error:
        load_error = error
    except Exception as error:  # noqa: BLE001 - handle corrupt/incompatible artifacts
        load_error = error

    render_sidebar(artifact)

    if artifact is None:
        if isinstance(load_error, FileNotFoundError):
            render_missing_model("missing")
        else:
            safe_detail = str(load_error).replace(str(PROJECT_DIR), ".") if load_error else ""
            render_missing_model("invalid", safe_detail)
        render_footer()
        return

    page = st.session_state.page
    page_changed = st.session_state.get("_rendered_page") != page
    st.session_state["_rendered_page"] = page

    if page == "Overview":
        render_overview(artifact)
    elif page == "Risk Analysis":
        render_risk_analysis(artifact)
    elif page == "Dataset Explorer":
        render_dataset_explorer()
    elif page == "Model Performance":
        render_model_performance(artifact)
    elif page == "Model Details":
        render_model_details(artifact)

    render_footer()

    if page_changed:
        scroll_main_to_top()


if __name__ == "__main__":
    main()
