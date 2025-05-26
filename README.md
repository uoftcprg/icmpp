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

Evaluate the (MC)ICM and a baseline algorithm.

```console
python scripts/analyze.py < scripts/data.npz > scripts/analyze.csv
```

Perform statistical testing.

```console
python scripts/experiment.py < scripts/analyze.csv > scripts/experiment.json
python scripts/experiment2.py < scripts/analyze.csv > scripts/experiment2.json
```

Plot results.

```console
python scripts/plot.py figures/results.pdf < scripts/analyze.csv
python scripts/plot2.py figures/results2.pdf < scripts/analyze.csv
```
