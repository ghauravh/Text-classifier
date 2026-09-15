# Text Classifier

A learning project that classifies messages as `ham` or `spam` with TF-IDF features and either logistic regression or a small neural network.

## Quick Start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py app.py train
py app.py predict --model artifacts/logistic_model.joblib --text "Claim your free reward now"
```

Train with your own UTF-8 CSV by providing `text` and `label` columns:

```powershell
py app.py train --data path\to\messages.csv --output-dir artifacts
```

Run the tests with:

```powershell
py -m unittest discover -s tests -v
```

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the complete step-by-step explanation.