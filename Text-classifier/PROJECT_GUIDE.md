# Text Classifier: Step-by-Step Guide

## 1. What This Project Does

This project learns patterns that distinguish ordinary messages (`ham`) from unwanted messages (`spam`). It trains two models so their strengths can be compared:

1. **Logistic regression** is a fast and interpretable linear baseline.
2. **Small neural network** has one hidden layer with 32 neurons and can learn nonlinear feature combinations.

Both models use exactly the same data split and TF-IDF features. The command-line app can train them or load a saved model to classify new text with confidence scores.

## 2. Project Files

- `app.py`: cleaning, splitting, training, evaluation, saving, and prediction code.
- `data/messages.csv`: a small balanced learning dataset with `text` and `label` columns.
- `tests/test_app.py`: checks text cleaning, data splitting, and probability output.
- `requirements.txt`: Python package dependencies.
- `artifacts/`: created during training; contains models and experiment history.

## 3. Set Up Python

Create a virtual environment so this project's packages are isolated:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## 4. Understand the Data

Open `data/messages.csv`. Each row contains:

- `text`: the message supplied to the model.
- `label`: the correct category (`ham` or `spam`).

The included data is intentionally small and synthetic. It makes the workflow easy to run, but it is not representative enough for a production spam filter. A serious model needs more diverse, carefully reviewed examples.

## 5. Clean the Text

`clean_text()` in `app.py` lowercases text, removes leading/trailing whitespace, and collapses repeated whitespace. `load_dataset()` then removes missing, empty, and duplicate rows.

Cleaning reduces meaningless variation. Importantly, the same `clean_text()` function is applied to both training examples and new predictions.

## 6. Split the Dataset

`split_dataset()` creates three partitions:

- **Training set (60%)**: fits the TF-IDF vocabulary and model weights.
- **Validation set (20%)**: compares model choices and reveals overfitting.
- **Test set (20%)**: gives a final estimate on untouched examples.

The split is *stratified*, so each partition keeps approximately the same proportion of ham and spam. `random_state=42` makes experiments reproducible.

Never train on validation or test examples. Doing so leaks answers into the model and makes metrics misleading.

## 7. Convert Text with TF-IDF

Machine-learning models need numbers rather than raw strings. `TfidfVectorizer` creates one feature for each observed word or two-word phrase.

**Term frequency (TF)** increases when a term appears more in a message. **Inverse document frequency (IDF)** reduces the weight of terms that occur in many messages. A feature is roughly:

$$
\operatorname{tfidf}(t,d) = \operatorname{tf}(t,d) \times \log\left(\frac{N}{\operatorname{df}(t)}\right)
$$

Words such as "prize" and phrases such as "claim now" can therefore become useful spam signals, while common words receive less influence.

## 8. Train Both Models

Call the reusable training function:

```python
from pathlib import Path
from app import train_models

train_models(Path("data/messages.csv"), Path("artifacts"))
```

`build_model()` creates a scikit-learn `Pipeline`. The pipeline first transforms text with TF-IDF and then fits one classifier:

- `LogisticRegression` estimates class probabilities from a weighted sum of features.
- `MLPClassifier` passes features through a 32-neuron hidden layer before producing probabilities.

Keeping transformation and classification in one pipeline prevents training/prediction preprocessing differences.

## 9. Understand Regularization and Overfitting

**Overfitting** occurs when a model memorizes training examples but performs poorly on unseen messages. Compare the printed training and validation F1 scores. A high training score with a much lower validation score is a warning.

Regularization discourages overly complex weights:

- Logistic regression uses L2 regularization controlled by `C`. Lower `C` means stronger regularization.
- The neural network uses L2 regularization controlled by `alpha`. Higher `alpha` means stronger regularization.

Change one parameter at a time, retrain, and compare validation results. Do not choose settings based on the test score.

## 10. Read the Metrics

- **Precision**: among messages predicted as a class, how many were correct?
- **Recall**: among actual messages of a class, how many did the model find?
- **F1 score**: harmonic mean of precision and recall.

$$
F_1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

The app reports **macro F1**, which calculates F1 for each class and averages them equally. This is useful when class sizes differ. Accuracy alone can hide failure on a minority class.

## 11. Track Experiments

Every training run appends one JSON object per model to `artifacts/experiments.jsonl`. Each record includes:

- timestamp and random seed;
- dataset path and row count;
- training, validation, and test metrics;
- saved model path.

JSON Lines keeps each experiment on its own line and can be loaded later with pandas. When experimenting, record the parameter change in your notes and compare validation F1 across runs.

## 12. Classify New Text

After training, call the reusable prediction function:

```python
from pathlib import Path
from app import predict_text_result

result = predict_text_result(
	Path("artifacts/logistic_model.joblib"),
	"Claim your free cash prize now",
)
print(result)
```

The output shows the predicted label and probability for every class. A confidence score is the model's estimated probability, not a guarantee that the answer is correct. Confidence can be poorly calibrated, especially with this tiny dataset.

To use the neural network instead, select `artifacts/neural_model.joblib`.

## 13. Use the FastAPI Service

Install the API dependencies, then start the server from the project folder:

```powershell
py -m pip install -r requirements.txt
py -m uvicorn api:app --reload
```

Send user text to `http://127.0.0.1:8000/predict` as JSON:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
	-ContentType "application/json" `
	-Body '{"text":"Claim your free reward now","model":"logistic"}'
```

The response includes the predicted label, winning confidence, and confidence scores for every label. Open `http://127.0.0.1:8000/docs` for the interactive API page.

## 14. Use Your Own Dataset

Create a UTF-8 CSV with this shape:

```csv
text,label
Are we still meeting at five,ham
Click now to collect your reward,spam
```

Then pass the dataset path to `train_models()`:

```python
from pathlib import Path
from app import train_models

train_models(Path("path/to/your_data.csv"), Path("artifacts"))
```

Use many examples from the real problem domain, check labels carefully, remove sensitive information, and confirm that duplicates do not cross partitions.

## 15. Suggested Experiments

1. Change logistic regression `C` to `0.1` and `10.0`.
2. Change neural-network `alpha` to `0.0001` and `0.01`.
3. Compare unigrams only (`ngram_range=(1, 1)`) with unigrams plus bigrams.
4. Add realistic examples and observe which metrics change.
5. Inspect false positives and false negatives instead of judging only one score.

Only use validation performance to select an experiment. Once selected, report its test result as the final unbiased estimate.
