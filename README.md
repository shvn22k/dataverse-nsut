# Malware Detection - Dataverse NSUT

Android malware detection using ensemble gradient boosting models.

## Project Structure

```
dataverse-nsut/
├── eda-and-exps/          # EDA notebooks and experiments
├── submission-1/          # First submission files
└── data/                  # Dataset files (not included in repo)
```

## Models

- **XGBoost** - Gradient boosting with decision trees
- **LightGBM** - Light gradient boosting machine
- **Ensemble** - Weighted average of both models


## Setup

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

2. Place data files in `data/` directory
3. Run the training script

## License

MIT

