# icmpp

This work explores different methodologies of assessing the utility value of poker tournaments.

## Instructions

Set the project root directory as a Python path.

```console
export PYTHONPATH=.
```

Preprocess the poker tournament dataset to fetch chip counts and results.

```console
python scripts/wsop.com.py scripts/pt-dataset/wsop.com/data/results.csv scripts/pt-dataset/wsop.com/data/chipcounts.csv > scripts/wsop.com.json
python scripts/pokernews.py scripts/pt-dataset/pokernews/data/chips.csv scripts/pt-dataset/pokernews/data/payouts.csv > scripts/pokernews.json
```

Combine, deduplicate, normalize, and pad the WSOP dataset.

```console
python scripts/data.py scripts/wsop.com.json scripts/pokernews.json > scripts/data.npz
```

Evaluate baselines.

```console
python scripts/baselines.py < scripts/data.npz > scripts/baselines.json
```

Train an AI model.

```console
python scripts/learn.py sklearn.linear_model.LinearRegression < scripts/data.npz > scripts/learn.pt
```

Plot results.

```console
python scripts/plot.py scripts/baselines.json scripts/learn.pt
```
