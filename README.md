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

Citing
------

If you use our code in your research, please cite the following:

```bibtex
@INPROCEEDINGS{11114139,
  author={Kim, Juho},
  booktitle={2025 IEEE Conference on Games (CoG)}, 
  title={Empirical Validation of the Independent Chip Model}, 
  year={2025},
  volume={},
  number={},
  pages={1-4},
  keywords={Video games;Games;Organizations;Multi-agent systems;Card games;Games of chance;Multi-agent systems;Poker;Strategy games},
  doi={10.1109/CoG64752.2025.11114139}}
```
