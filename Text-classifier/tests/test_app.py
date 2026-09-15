"""Focused tests for data preparation and model behavior."""

import unittest

import pandas as pd

from app import build_model, clean_text, split_dataset


class TextClassifierTests(unittest.TestCase):
    """Verify the most important reusable parts of the workflow."""

    def test_clean_text_normalizes_case_and_whitespace(self) -> None:
        # The same normalization must happen during training and prediction.
        self.assertEqual(clean_text("  Hello\n  WORLD  "), "hello world")

    def test_split_dataset_creates_expected_partitions(self) -> None:
        # Twenty balanced rows are enough to verify the 60/20/20 split exactly.
        dataset = pd.DataFrame(
            {
                "text": [f"ham message {index}" for index in range(10)]
                + [f"spam message {index}" for index in range(10)],
                "label": ["ham"] * 10 + ["spam"] * 10,
            }
        )
        train_set, validation_set, test_set = split_dataset(dataset)
        self.assertEqual((len(train_set), len(validation_set), len(test_set)), (12, 4, 4))
        self.assertEqual(validation_set["label"].nunique(), 2)
        self.assertEqual(test_set["label"].nunique(), 2)

    def test_logistic_pipeline_returns_probabilities(self) -> None:
        # A tiny fit proves that TF-IDF and the classifier work together end to end.
        model = build_model("logistic")
        model.fit(
            ["team meeting tomorrow", "family dinner tonight", "win cash now", "claim free prize"],
            ["ham", "ham", "spam", "spam"],
        )
        probabilities = model.predict_proba(["claim cash prize"])[0]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertEqual(len(probabilities), 2)


if __name__ == "__main__":
    unittest.main()