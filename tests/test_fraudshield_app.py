"""Focused, read-only tests for the FraudShield AI dashboard.

Run from the project folder with:

    python -B -m unittest discover -s tests -v

The tests never train a model, never write to the real model/dataset, and use
temporary directories for all failure scenarios.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - Streamlit is a project dependency
    AppTest = None


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "10_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("fraudshield_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load_app_module()

FEATURES = [f"V{i}" for i in range(1, 31)]
DATASET_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


class FakeModel:
    """Deterministic classifier that records the exact frame it receives."""

    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.columns = None
        self.shape = None

    def predict_proba(self, frame):
        self.columns = list(frame.columns)
        self.shape = frame.shape
        probability = float(self.probabilities.reshape(-1)[0])
        return np.array([[1.0 - probability, probability]])

    def predict(self, frame):
        probability = float(self.probabilities.reshape(-1)[0])
        return np.array([int(probability >= 0.5)])


class ValidModel(FakeModel):
    n_features_in_ = 30
    feature_names_in_ = np.array(FEATURES)


class WrongCountModel(FakeModel):
    n_features_in_ = 31


class PermutedFeatureModel(ValidModel):
    feature_names_in_ = np.array(FEATURES[::-1])


class UnexpectedClassOrderModel(ValidModel):
    classes_ = np.array([1, 0])


def feature_values():
    return {name: float(index) for index, name in enumerate(FEATURES)}


def app_text(at):
    """Collect the human-visible text emitted by AppTest."""
    pieces = []
    for attribute in ("markdown", "caption", "text", "warning", "error", "info", "success"):
        for element in getattr(at, attribute, []):
            value = getattr(element, "value", None)
            if value:
                pieces.append(str(value))
    return "\n".join(pieces)


def make_temp_app(with_model=True, dataset=None, corrupt_model=False, artifact_threshold=None):
    """Copy the app into an isolated temp directory; never touch real files."""
    temp_dir = tempfile.mkdtemp(prefix="fraudshield_apptest_")
    directory = Path(temp_dir)
    (directory / "models").mkdir()
    shutil.copy2(APP_PATH, directory / "10_app.py")
    if with_model:
        model_target = directory / "models" / "fraudshield_random_forest.joblib"
        if corrupt_model:
            model_target.write_bytes(b"not a valid joblib artifact")
        elif artifact_threshold is None:
            shutil.copy2(app.MODEL_PATH, model_target)
        else:
            modified = app.joblib.load(app.MODEL_PATH)
            modified["threshold"] = float(artifact_threshold)
            app.joblib.dump(modified, model_target)
    if dataset is not None:
        (directory / "creditcard.csv").write_text(dataset)
    return directory


class TestArtifactAndMetrics(unittest.TestCase):
    def test_saved_artifact_contract(self):
        artifact = app.load_model_artifact(str(app.MODEL_PATH))
        self.assertEqual(artifact["threshold"], 0.20)
        self.assertEqual(type(artifact["model"]).__name__, "RandomForestClassifier")
        self.assertEqual(len(artifact["feature_names"]), 30)
        self.assertNotIn("Class", artifact["feature_names"])
        self.assertEqual(artifact["feature_names"][0], "Time")
        self.assertEqual(artifact["feature_names"][-1], "Amount")
        self.assertEqual(list(artifact["model"].feature_names_in_), artifact["feature_names"])
        self.assertEqual(artifact["model"].n_features_in_, 30)
        self.assertEqual(list(artifact["model"].classes_), [0, 1])
        self.assertEqual(artifact["model"].n_estimators, 100)
        self.assertEqual(artifact["model"].class_weight, "balanced")

    def test_saved_model_scores_first_dataset_row_end_to_end(self):
        if not app.DATASET_PATH.exists():
            self.skipTest("creditcard.csv is not present")
        row = pd.read_csv(app.DATASET_PATH, nrows=1).iloc[0]
        values = {name: float(row[name]) for name in app.load_model_artifact(str(app.MODEL_PATH))["feature_names"]}
        artifact = app.load_model_artifact(str(app.MODEL_PATH))
        probability, prediction = app.predict_transaction(
            artifact["model"], artifact["feature_names"], artifact["threshold"], values
        )
        self.assertEqual(probability, 0.0)
        self.assertEqual(prediction, 0)
        self.assertEqual(int(row["Class"]), 0)

    def test_reference_metrics_constants_are_internally_consistent(self):
        results = app.TEST_RESULTS
        self.assertEqual(results["precision"], 0.8261)
        self.assertEqual(results["recall"], 0.8028)
        self.assertEqual(results["f1"], 0.8143)
        self.assertEqual(results["average_precision"], 0.7967)
        self.assertEqual(
            (results["tn"], results["fp"], results["fn"], results["tp"]),
            (42_476, 12, 14, 57),
        )
        self.assertEqual(results["transactions"], 42_559)
        self.assertEqual(results["actual_fraud"], 71)
        self.assertEqual(results["actual_legitimate"], 42_488)
        self.assertEqual(results["tn"] + results["fp"], results["actual_legitimate"])
        self.assertEqual(results["tp"] + results["fn"], results["actual_fraud"])
        self.assertEqual(
            results["tn"] + results["fp"] + results["fn"] + results["tp"],
            results["transactions"],
        )
        computed_precision = results["tp"] / (results["tp"] + results["fp"])
        computed_recall = results["tp"] / (results["tp"] + results["fn"])
        computed_f1 = 2 * computed_precision * computed_recall / (
            computed_precision + computed_recall
        )
        self.assertAlmostEqual(computed_precision, results["precision"], places=4)
        self.assertAlmostEqual(computed_recall, results["recall"], places=4)
        self.assertAlmostEqual(computed_f1, results["f1"], places=4)

    def test_evaluation_threshold_is_explicitly_historical(self):
        self.assertEqual(app.EVALUATION_THRESHOLD, 0.20)
        self.assertEqual(app.TEST_RESULTS["threshold"], 0.20)
        self.assertEqual(app.evaluation_threshold_text(), "0.20")

    def test_live_artifact_threshold_is_reported_from_the_artifact(self):
        artifact = app.load_model_artifact(str(app.MODEL_PATH))
        self.assertEqual(app.artifact_threshold_text(artifact), "0.20")
        self.assertIsNone(app.threshold_mismatch_message(artifact))

    def test_threshold_mismatch_explains_without_recalculating_metrics(self):
        artifact = {"threshold": 0.35}
        message = app.threshold_mismatch_message(artifact)
        self.assertIsNotNone(message)
        self.assertIn("0.35", message)
        self.assertIn("0.20", message)
        self.assertIn("not recalculated", message)
        self.assertIn("Live predictions use 0.35", message)
        self.assertEqual(app.artifact_threshold_text(artifact), "0.35")

    def test_metric_definition_details_match_reference_counts(self):
        definitions = {name: detail for name, _, detail in app.METRIC_DEFINITIONS}
        self.assertIn("57 true positives / 69 fraud alerts", definitions["Precision"])
        self.assertIn("57 detected / 71 actual frauds", definitions["Recall"])
        self.assertIn("0.8143", definitions["F1-score"])
        self.assertIn("0.7967", definitions["Average Precision"])

    def test_accuracy_is_not_presented_as_perfect(self):
        results = app.TEST_RESULTS
        accuracy = (results["tn"] + results["tp"]) / results["transactions"]
        baseline = (results["tn"] + results["fp"]) / results["transactions"]
        self.assertAlmostEqual(accuracy, (42_476 + 57) / 42_559, places=7)
        self.assertAlmostEqual(baseline, (42_476 + 12) / 42_559, places=7)
        self.assertLess(accuracy, 1.0)
        self.assertLess(baseline, accuracy)


class TestInference(unittest.TestCase):
    def test_probability_below_equal_above_threshold(self):
        for probability, expected in ((0.1999, 0), (0.2000, 1), (0.2001, 1)):
            with self.subTest(probability=probability):
                model = FakeModel(probability)
                observed_probability, prediction = app.predict_transaction(
                    model, FEATURES, 0.20, feature_values()
                )
                self.assertEqual(observed_probability, probability)
                self.assertEqual(prediction, expected)

    def test_second_probability_column_is_used(self):
        class TwoColumnModel:
            classes_ = np.array([0, 1])

            def predict_proba(self, frame):
                return np.array([[0.90, 0.10]])

        probability, prediction = app.predict_transaction(
            TwoColumnModel(), FEATURES, 0.20, feature_values()
        )
        self.assertEqual(probability, 0.10)
        self.assertEqual(prediction, 0)

    def test_fraud_probability_column_follows_class_label_one(self):
        class ReversedClassModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, frame):
                return np.array([[0.80, 0.20]])

        probability, prediction = app.predict_transaction(
            ReversedClassModel(), FEATURES, 0.50, feature_values()
        )
        self.assertEqual(probability, 0.80)
        self.assertEqual(prediction, 1)

    def test_model_without_classes_is_rejected_for_inference(self):
        class NoClassesModel:
            def predict_proba(self, frame):
                return np.array([[0.90, 0.10]])

        with self.assertRaisesRegex(ValueError, "does not expose classes_"):
            app.predict_transaction(
                NoClassesModel(), FEATURES, 0.20, feature_values()
            )

    def test_threshold_argument_is_used_not_a_hardcoded_constant(self):
        for threshold, expected in ((0.20, 1), (0.50, 1), (0.80, 0)):
            with self.subTest(threshold=threshold):
                _, prediction = app.predict_transaction(
                    FakeModel(0.50), FEATURES, threshold, feature_values()
                )
                self.assertEqual(prediction, expected)

    def test_feature_order_is_restored_and_extra_keys_are_ignored(self):
        model = FakeModel(0.50)
        reversed_values = dict(reversed(list(feature_values().items())))
        reversed_values["Class"] = 1
        reversed_values["not_a_model_feature"] = 99
        _, prediction = app.predict_transaction(model, FEATURES, 0.20, reversed_values)
        self.assertEqual(prediction, 1)
        self.assertEqual(model.shape, (1, 30))
        self.assertEqual(model.columns, FEATURES)
        self.assertNotIn("Class", model.columns)
        self.assertNotIn("not_a_model_feature", model.columns)

    def test_missing_feature_is_rejected(self):
        values = feature_values()
        del values["V5"]
        with self.assertRaisesRegex(ValueError, "Missing model features: V5"):
            app.predict_transaction(FakeModel(0.5), FEATURES, 0.20, values)

    def test_nonnumeric_feature_is_rejected(self):
        values = feature_values()
        values["V5"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "valid numeric value"):
            app.predict_transaction(FakeModel(0.5), FEATURES, 0.20, values)

    def test_nan_feature_is_rejected(self):
        values = feature_values()
        values["V5"] = float("nan")
        with self.assertRaisesRegex(ValueError, "valid numeric value"):
            app.predict_transaction(FakeModel(0.5), FEATURES, 0.20, values)

    def test_infinite_feature_is_rejected(self):
        values = feature_values()
        values["V5"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            app.predict_transaction(FakeModel(0.5), FEATURES, 0.20, values)

    def test_fewer_feature_names_are_rejected_by_artifact_validation(self):
        artifact = {
            "model": ValidModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES[:29],
        }
        with self.assertRaisesRegex(ValueError, "Expected 30 model features"):
            app._validate_artifact(artifact)


class TestValidationAndPresentation(unittest.TestCase):
    def test_valid_artifact_is_accepted(self):
        artifact = {
            "model": ValidModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES,
        }
        validated = app._validate_artifact(artifact)
        self.assertEqual(validated["threshold"], 0.20)
        self.assertEqual(validated["feature_names"], FEATURES)

    def test_invalid_artifacts_are_rejected(self):
        valid = {
            "model": ValidModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES,
        }
        cases = {
            "not a dictionary": [],
            "missing key": {"model": ValidModel(0.5), "threshold": 0.20},
            "non-numeric threshold": {**valid, "threshold": "abc"},
            "threshold above one": {**valid, "threshold": 1.01},
            "threshold below zero": {**valid, "threshold": -0.01},
            "29 features": {**valid, "feature_names": FEATURES[:29]},
            "duplicate features": {**valid, "feature_names": FEATURES[:29] + ["V1"]},
            "invalid feature name": {**valid, "feature_names": FEATURES[:29] + [None]},
            "model feature-count mismatch": {**valid, "model": WrongCountModel(0.5)},
            "object without predict_proba": {**valid, "model": object()},
        }
        for label, artifact in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    app._validate_artifact(artifact)

    def test_permuted_feature_names_are_rejected(self):
        artifact = {
            "model": PermutedFeatureModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES,
        }
        with self.assertRaisesRegex(ValueError, "feature names or order"):
            app._validate_artifact(artifact)

    def test_unexpected_class_ordering_is_rejected(self):
        artifact = {
            "model": UnexpectedClassOrderModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES,
        }
        with self.assertRaisesRegex(ValueError, "classes_"):
            app._validate_artifact(artifact)

    def test_model_without_declared_classes_is_rejected(self):
        class NoClassesModel(FakeModel):
            n_features_in_ = 30
            feature_names_in_ = np.array(FEATURES)
            classes_ = None

        artifact = {
            "model": NoClassesModel(0.5),
            "threshold": 0.20,
            "feature_names": FEATURES,
        }
        with self.assertRaisesRegex(ValueError, "does not expose classes_"):
            app._validate_artifact(artifact)

    def test_confusion_matrix_orientation_and_labels(self):
        markup = app.confusion_matrix_html(42_476, 12, 14, 57)
        self.assertIn("Actual", markup)
        self.assertIn("Predicted<br/>legitimate", markup)
        self.assertIn("Predicted<br/>fraud", markup)
        self.assertLess(markup.index("True negatives"), markup.index("False positives"))
        self.assertLess(markup.index("False positives"), markup.index("False negatives"))
        self.assertLess(markup.index("False negatives"), markup.index("True positives"))
        self.assertLess(markup.index("42,476"), markup.index("12"))
        self.assertLess(markup.index("12"), markup.index("14"))
        self.assertLess(markup.index("14"), markup.index("57"))

    def test_gauge_marker_and_color_follow_threshold(self):
        below = app.gauge_html(0.1999, 0.20)
        equal = app.gauge_html(0.2000, 0.20)
        above = app.gauge_html(0.2001, 0.20)
        self.assertNotIn("fs-gauge-fill danger", below)
        self.assertIn("width:19.990%", below)
        self.assertIn("left:20.000%", below)
        self.assertIn("fs-gauge-fill danger", equal)
        self.assertIn("width:20.000%", equal)
        self.assertIn("left:20.000%", equal)
        self.assertIn("fs-gauge-fill danger", above)
        self.assertIn("width:20.010%", above)
        self.assertIn("left:20.000%", above)

    def test_probability_formatting(self):
        self.assertEqual(app.fmt_probability(0.00001), "&lt;0.01%")
        self.assertEqual(app.fmt_probability(0.001), "0.10%")
        self.assertEqual(app.fmt_probability(0.20), "20.00%")


class TestSafeLoadingAndCaching(unittest.TestCase):
    def test_load_dataset_summary_on_synthetic_data(self):
        rows = [
            [0] + [0.0] * 28 + [10.0, 0],
            [1] + [1.0] + [0.0] * 27 + [20.0, 1],
            [1] + [1.0] + [0.0] * 27 + [20.0, 1],  # exact duplicate
            [2] + [float("nan")] + [0.0] * 27 + [30.0, 0],
        ]
        with tempfile.TemporaryDirectory(prefix="fraudshield_data_") as temp_dir:
            path = Path(temp_dir) / "creditcard.csv"
            pd.DataFrame(rows, columns=DATASET_COLUMNS).to_csv(path, index=False)
            data, summary = app.load_dataset(str(path))
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["columns"], 31)
        self.assertEqual(summary["fraud_count"], 2)
        self.assertEqual(summary["legitimate_count"], 2)
        self.assertEqual(summary["fraud_rate"], 0.5)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["deduplicated_rows"], 3)
        self.assertEqual(summary["missing_total"], 1)
        self.assertEqual(len(data), 4)

    def test_load_dataset_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            app.load_dataset("/tmp/opencode/fraudshield_does_not_exist.csv")

    def test_load_dataset_missing_class_column(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_data_") as temp_dir:
            path = Path(temp_dir) / "creditcard.csv"
            pd.DataFrame(
                [[0] + [0.0] * 28 + [10.0]], columns=DATASET_COLUMNS[:-1]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "required 'Class' column"):
                app.load_dataset(str(path))

    def test_load_dataset_empty_file_returns_zero_summary(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_data_") as temp_dir:
            path = Path(temp_dir) / "creditcard.csv"
            pd.DataFrame(columns=DATASET_COLUMNS).to_csv(path, index=False)
            _, summary = app.load_dataset(str(path))
        self.assertEqual(summary["rows"], 0)
        self.assertEqual(summary["fraud_count"], 0)
        self.assertEqual(summary["fraud_rate"], 0.0)

    def test_load_dataset_zero_byte_file_returns_zero_summary(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_data_") as temp_dir:
            path = Path(temp_dir) / "creditcard.csv"
            path.write_text("")
            data, summary = app.load_dataset(str(path))
        self.assertEqual(summary["rows"], 0)
        self.assertEqual(summary["columns"], 31)
        self.assertIn("Class", data.columns)
        self.assertEqual(summary["fraud_count"], 0)
        self.assertEqual(summary["fraud_rate"], 0.0)

    def test_missing_model_artifact(self):
        with self.assertRaises(FileNotFoundError):
            app.load_model_artifact("/tmp/opencode/fraudshield_does_not_exist.joblib")

    def test_corrupt_model_artifact(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_model_") as temp_dir:
            path = Path(temp_dir) / "bad.joblib"
            path.write_bytes(b"not a joblib artifact")
            with self.assertRaises(Exception):
                app.load_model_artifact(str(path))

    def test_model_is_loaded_once_per_cached_path(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_model_") as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            shutil.copy2(app.MODEL_PATH, path)
            with mock.patch.object(app.joblib, "load", wraps=app.joblib.load) as loader:
                first = app.load_model_artifact(str(path))
                second = app.load_model_artifact(str(path))
            self.assertEqual(loader.call_count, 1)
            self.assertIs(first, second)

    def test_dataset_csv_is_read_once_per_cached_path(self):
        with tempfile.TemporaryDirectory(prefix="fraudshield_data_") as temp_dir:
            path = Path(temp_dir) / "creditcard.csv"
            pd.DataFrame(
                [[0] + [0.0] * 28 + [1.0, 0]], columns=DATASET_COLUMNS
            ).to_csv(path, index=False)
            with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as reader:
                first_data, first_summary = app.load_dataset(str(path))
                second_data, second_summary = app.load_dataset(str(path))
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(first_summary, second_summary)
            self.assertTrue(first_data.equals(second_data))

    def test_app_paths_are_project_relative(self):
        self.assertEqual(app.PROJECT_DIR, ROOT)
        self.assertEqual(app.MODEL_PATH, ROOT / "models" / "fraudshield_random_forest.joblib")
        self.assertEqual(app.DATASET_PATH, ROOT / "creditcard.csv")

    def test_app_source_has_no_training_or_model_writes(self):
        source = APP_PATH.read_text()
        self.assertNotIn(".fit(", source)
        self.assertNotIn("joblib.dump", source)
        self.assertIn("joblib.load", source)
        for token in ("import requests", "import urllib", "import httpx", "import aiohttp", "http://"):
            self.assertNotIn(token, source)
        https_urls = re.findall(r"https://[^\s\"'<>]+", source)
        self.assertEqual(
            https_urls,
            ["https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"],
        )


@unittest.skipUnless(AppTest is not None, "Streamlit AppTest is unavailable")
class TestDashboardIntegration(unittest.TestCase):
    def test_overview_and_all_navigation_pages_render(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        self.assertEqual(at.exception, [])
        overview = app_text(at)
        for expected in ("82.61%", "80.28%", "0.8143", "0.7967", "42,559", "71"):
            self.assertIn(expected, overview)

        at.button(key="nav_Model Performance").click().run()
        self.assertEqual(at.exception, [])
        performance = app_text(at)
        for expected in ("99.94%", "99.83%", "42,476", "False positives", "False negatives"):
            self.assertIn(expected, performance)

        at.button(key="nav_Model Details").click().run()
        self.assertEqual(at.exception, [])
        details = app_text(at)
        self.assertIn("models/fraudshield_random_forest.joblib", details)
        self.assertIn("Time", details)
        self.assertIn("Amount", details)

        at.button(key="nav_Dataset Explorer").click().run()
        self.assertEqual(at.exception, [])
        explorer = app_text(at)
        for expected in ("284,807", "31", "492", "1,081", "283,726"):
            self.assertIn(expected, explorer)

        at.button(key="nav_Risk Analysis").click().run()
        self.assertEqual(at.exception, [])
        risk = app_text(at)
        self.assertIn("Dataset row", risk)
        self.assertIn("Dataset label", risk)
        self.assertIn("Model estimate", risk)
        input_table = at.dataframe[-1].value
        self.assertEqual(input_table.shape, (30, 3))
        self.assertNotIn("Class", set(input_table["Feature"]))

    def test_sample_mode_random_fraud_shows_dataset_label_separately(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        at.button(key="nav_Risk Analysis").click().run()
        for button in at.button:
            if button.label == "Random fraud":
                button.click().run()
                break
        self.assertEqual(at.exception, [])
        facts = [str(element.value) for element in at.markdown if "fs-facts" in str(element.value)]
        truth = [str(element.value) for element in at.markdown if "fs-truth-value" in str(element.value)]
        self.assertTrue(any("Dataset label" in block and "Fraud" in block for block in facts))
        self.assertTrue(any("fs-truth-value danger" in block for block in truth))

    def test_custom_mode_rejects_empty_vector(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        at.button(key="nav_Risk Analysis").click().run()
        at.radio[0].set_value("Custom feature input").run()
        for button in at.button:
            if button.label == "Analyze transaction":
                button.click().run()
                break
        self.assertEqual(at.exception, [])
        self.assertIn("30 of 30 required features are empty.", app_text(at))

    def test_custom_mode_scores_complete_30_feature_vector(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        at.button(key="nav_Risk Analysis").click().run()
        at.radio[0].set_value("Custom feature input").run()
        for key in list(at.session_state):
            if key.startswith("custom_feature_"):
                at.session_state[key] = 0.0
        at.run()
        for button in at.button:
            if button.label == "Analyze transaction":
                button.click().run()
                break
        self.assertEqual(at.exception, [])
        text = app_text(at)
        self.assertIn("30 / 30", text)
        self.assertIn("Predicted legitimate", text)
        self.assertIn("20.00%", text)

    def test_footer_dataset_source_is_complete_and_links_to_kaggle(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        self.assertEqual(at.exception, [])
        text = app_text(at)
        self.assertIn(
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", text
        )
        self.assertIn("Dataset Source", text)
        self.assertIn("Credit Card Fraud Detection", text)
        self.assertIn("284,807", text)
        self.assertIn("31 columns", text)
        self.assertIn("Class = 1", text)
        self.assertIn("Class = 0", text)
        self.assertIn("V1", text)
        self.assertIn("V28", text)
        self.assertIn("Time", text)
        self.assertIn("Amount", text)
        self.assertIn("duplicate rows", text)
        self.assertIn("does not download data from Kaggle", text)

    def test_threshold_labels_separate_live_artifact_and_evaluation(self):
        at = AppTest.from_file(str(APP_PATH), default_timeout=180)
        at.run()
        overview = app_text(at)
        self.assertIn("Artifact threshold 0.20", overview)
        self.assertIn("evaluation threshold 0.20", overview)

        at.button(key="nav_Model Details").click().run()
        self.assertEqual(at.exception, [])
        details = app_text(at)
        self.assertIn("Live artifact threshold", details)
        self.assertIn("Evaluation threshold", details)
        self.assertIn("0.20 evaluation threshold", details)

    def test_threshold_mismatch_is_explained_on_metric_pages(self):
        directory = make_temp_app(with_model=True, artifact_threshold=0.35)
        try:
            at = AppTest.from_file(str(directory / "10_app.py"), default_timeout=120)
            at.run()
            self.assertEqual(at.exception, [])
            overview = app_text(at)
            self.assertIn("decision threshold of 0.35", overview)
            self.assertIn("evaluated at 0.20", overview)
            self.assertIn("not recalculated", overview)
            self.assertIn("82.61%", overview)

            at.button(key="nav_Model Performance").click().run()
            self.assertEqual(at.exception, [])
            performance = app_text(at)
            self.assertIn("decision threshold of 0.35", performance)
            self.assertIn("evaluated at 0.20", performance)
            self.assertIn("82.61%", performance)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_missing_model_shows_safe_page(self):
        directory = make_temp_app(with_model=False)
        try:
            at = AppTest.from_file(str(directory / "10_app.py"), default_timeout=120)
            at.run()
            self.assertEqual(at.exception, [])
            self.assertIn("saved model could not be loaded", app_text(at))
            self.assertEqual(at.button, [])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_missing_dataset_keeps_overview_and_performance_available(self):
        directory = make_temp_app(with_model=True)
        try:
            at = AppTest.from_file(str(directory / "10_app.py"), default_timeout=120)
            at.run()
            self.assertEqual(at.exception, [])
            self.assertIn("82.61%", app_text(at))

            at.button(key="nav_Dataset Explorer").click().run()
            self.assertEqual(at.exception, [])
            self.assertIn("creditcard.csv was not found", app_text(at))

            at.button(key="nav_Risk Analysis").click().run()
            self.assertEqual(at.exception, [])
            self.assertIn("creditcard.csv was not found", app_text(at))

            at.button(key="nav_Model Performance").click().run()
            self.assertEqual(at.exception, [])
            self.assertIn("82.61%", app_text(at))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_corrupt_model_shows_safe_page(self):
        directory = make_temp_app(with_model=True, corrupt_model=True)
        try:
            at = AppTest.from_file(str(directory / "10_app.py"), default_timeout=120)
            at.run()
            self.assertEqual(at.exception, [])
            self.assertIn("saved model could not be loaded", app_text(at))
            self.assertIn("Technical detail", [expander.label for expander in at.expander])
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TestRepositoryReadiness(unittest.TestCase):
    def test_gitignore_excludes_dataset_but_keeps_required_files(self):
        gitignore_path = ROOT / ".gitignore"
        self.assertTrue(gitignore_path.exists())
        lines = {
            line.strip()
            for line in gitignore_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        self.assertIn("creditcard.csv", lines)
        self.assertTrue((ROOT / "creditcard.csv").exists())
        self.assertNotIn("*.py", lines)
        self.assertNotIn("*.joblib", lines)
        for required in (
            "10_app.py",
            "README.md",
            "requirements.txt",
            "models/fraudshield_random_forest.joblib",
            "tests/test_fraudshield_app.py",
        ):
            self.assertNotIn(required, lines)

    def test_readme_documents_dataset_and_reproducible_setup(self):
        readme = (ROOT / "README.md").read_text()
        dataset_url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        self.assertIn(dataset_url, readme)
        for expected in (
            "284,807",
            "31 columns",
            "Class = 1",
            "Class = 0",
            "V1",
            "V28",
            "Time",
            "Amount",
            "drop_duplicates()",
            "82.61%",
            "80.28%",
            "0.8143",
            "0.7967",
            "pip install -r requirements.txt",
            "python -m streamlit run 10_app.py",
            "python -B -m unittest discover -s tests -v",
            "creditcard.csv",
        ):
            self.assertIn(expected, readme)
        self.assertIn("No public FraudShield AI deployment URL is currently confirmed", readme)
        self.assertIn("not a FraudShield AI deployment", readme)


class TestEmptyDataset(unittest.TestCase):
    @unittest.skipUnless(AppTest is not None, "Streamlit AppTest is unavailable")
    def test_empty_dataset_risk_analysis_does_not_crash(self):
        """An empty CSV must stop before row-dependent number inputs are built."""
        cases = {
            "header-only": pd.DataFrame(columns=DATASET_COLUMNS).to_csv(index=False),
            "zero-byte": "",
        }
        for label, empty_csv in cases.items():
            with self.subTest(label=label):
                directory = make_temp_app(with_model=True, dataset=empty_csv)
                try:
                    at = AppTest.from_file(
                        str(directory / "10_app.py"), default_timeout=120
                    )
                    at.run()
                    at.button(key="nav_Risk Analysis").click().run()
                    self.assertEqual(at.exception, [])
                    text = app_text(at)
                    self.assertIn(
                        "creditcard.csv contains no rows. "
                        "Sample-transaction mode is unavailable.",
                        text,
                    )
                    self.assertNotIn("Dataset row", text)
                finally:
                    shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
