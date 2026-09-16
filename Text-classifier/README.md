# Text Classifier

A learning project that classifies messages as `ham` or `spam` with TF-IDF features and either logistic regression or a small neural network.

## Quick Start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn api:app --reload
```

The API is available at `http://127.0.0.1:8000/docs`. It uses the trained model artifacts in `artifacts/` to classify messages.

Train models from Python with your own UTF-8 CSV by providing `text` and `label` columns:

```python
from pathlib import Path
from app import train_models

train_models(Path("path/to/messages.csv"), Path("artifacts"))
```

Run the tests with:

```powershell
py -m unittest discover -s tests -v
```

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the complete step-by-step explanation.
