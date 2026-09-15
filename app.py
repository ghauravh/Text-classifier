"""Train and use TF-IDF text classifiers from the command line."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_SEED = 42


def clean_text(text: str) -> str:
    """Normalize whitespace and lowercase text while preserving useful punctuation."""
    # Step 1: Convert every value to text so the cleaning pipeline is predictable.
    normalized = str(text).lower().strip()
    # Step 2: Replace repeated whitespace (spaces, tabs, and newlines) with one space.
    return re.sub(r"\s+", " ", normalized)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load, validate, clean, and deduplicate a text classification dataset."""
    # Step 1: Read the CSV. The project expects exactly the useful columns below.
    dataset = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    # Step 2: Remove rows with missing values because they cannot train the model.
    dataset = dataset.dropna(subset=["text", "label"]).copy()
    # Step 3: Apply the same cleaning function used for future predictions.
    dataset["text"] = dataset["text"].map(clean_text)
    dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()
    # Step 4: Remove empty and duplicate messages to reduce accidental data leakage.
    dataset = dataset[dataset["text"].str.len() > 0].drop_duplicates(subset=["text"])

    # Step 5: Require enough examples for stratified train/validation/test splits.
    class_counts = dataset["label"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 5:
        raise ValueError("Dataset needs at least two labels and five examples per label.")
    return dataset


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 60% training, 20% validation, and 20% test sets."""
    # Step 1: Hold out 40% while preserving each label's proportion.
    train_set, temporary_set = train_test_split(
        dataset,
        test_size=0.40,
        random_state=RANDOM_SEED,
        stratify=dataset["label"],
    )
    # Step 2: Divide the held-out data equally into validation and final test sets.
    validation_set, test_set = train_test_split(
        temporary_set,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_set["label"],
    )
    return train_set, validation_set, test_set


def build_model(model_name: str) -> Pipeline:
    """Build a TF-IDF pipeline with the selected classifier."""
    # Step 1: Convert words and two-word phrases into weighted numeric features.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

    # Step 2: Select either a linear baseline or a small feed-forward neural network.
    if model_name == "logistic":
        # Smaller C means stronger L2 regularization and less overfitting.
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    elif model_name == "neural":
        # One 32-neuron hidden layer keeps this network intentionally small.
        classifier = MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=0.001,
            max_iter=500,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Step 3: A pipeline guarantees identical TF-IDF processing in training and use.
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def calculate_metrics(model: Pipeline, dataset: pd.DataFrame) -> dict[str, Any]:
    """Calculate aggregate and per-class classification metrics."""
    # Step 1: Generate predictions for a dataset the model did not fit on here.
    predictions = model.predict(dataset["text"])
    # Step 2: Macro averaging gives every class equal importance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        dataset["label"], predictions, average="macro", zero_division=0
    )
    # Step 3: Keep both summary metrics and a detailed per-label report.
    return {
        "accuracy": float(accuracy_score(dataset["label"], predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            dataset["label"], predictions, output_dict=True, zero_division=0
        ),
    }


def train_models(data_path: Path, output_dir: Path) -> None:
    """Train, evaluate, save, and track both model types."""
    # Step 1: Prepare the data once so both models receive exactly the same examples.
    dataset = load_dataset(data_path)
    train_set, validation_set, test_set = split_dataset(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Rows: {len(dataset)} | train: {len(train_set)} | "
        f"validation: {len(validation_set)} | test: {len(test_set)}"
    )

    for model_name in ("logistic", "neural"):
        # Step 2: Fit TF-IDF and the classifier only on the training partition.
        model = build_model(model_name)
        model.fit(train_set["text"], train_set["label"])

        # Step 3: Compare train and validation scores to detect possible overfitting.
        train_metrics = calculate_metrics(model, train_set)
        validation_metrics = calculate_metrics(model, validation_set)
        # Step 4: Touch the test set only after model design decisions are complete.
        test_metrics = calculate_metrics(model, test_set)

        # Step 5: Persist the entire pipeline, including its fitted vocabulary.
        model_path = output_dir / f"{model_name}_model.joblib"
        joblib.dump(model, model_path)

        # Step 6: Append parameters and results as a lightweight experiment tracker.
        experiment = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "data_path": str(data_path),
            "dataset_rows": len(dataset),
            "random_seed": RANDOM_SEED,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "artifact": str(model_path),
        }
        with (output_dir / "experiments.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(experiment) + "\n")

        print(f"\n{model_name.upper()}")
        print(
            f"train F1={train_metrics['f1_macro']:.3f} | "
            f"validation F1={validation_metrics['f1_macro']:.3f} | "
            f"test F1={test_metrics['f1_macro']:.3f}"
        )
        print(f"Saved: {model_path}")


def predict_text(model_path: Path, text: str) -> None:
    """Load a fitted pipeline and classify one new message."""
    # Step 1: Restore the trained vectorizer and classifier together.
    model: Pipeline = joblib.load(model_path)
    cleaned_text = clean_text(text)
    # Step 2: predict_proba returns one confidence value for every known class.
    probabilities = model.predict_proba([cleaned_text])[0]
    classifier = model.named_steps["classifier"]
    classes = classifier.classes_
    scores = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)

    # Step 3: Show the winning label followed by all class confidence scores.
    print(f"Prediction: {scores[0][0]}")
    print("Confidence scores:")
    for label, probability in scores:
        print(f"  {label}: {probability:.2%}")


def parse_arguments() -> argparse.Namespace:
    """Define the train and predict command-line interfaces."""
    parser = argparse.ArgumentParser(description="Train and use a TF-IDF text classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate both models.")
    train_parser.add_argument("--data", type=Path, default=Path("data/messages.csv"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    predict_parser = subparsers.add_parser("predict", help="Classify one text message.")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--text", required=True)
    return parser.parse_args()


def main() -> None:
    """Route the selected command to its workflow."""
    arguments = parse_arguments()
    if arguments.command == "train":
        train_models(arguments.data, arguments.output_dir)
    else:
        predict_text(arguments.model, arguments.text)


if __name__ == "__main__":
    main()