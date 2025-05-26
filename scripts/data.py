from collections import defaultdict
from itertools import starmap
from json import load
from sys import argv, stdout

import numpy as np

from icmpp import baseline, normalize, pad_like

DATA_PATHS = argv[1:]


def main():
    data_set = set()
    data = defaultdict(list)

    for path in DATA_PATHS:
        with open(path) as file:
            sub_data = load(file)

            for chip_counts, places, payouts in zip(
                    sub_data['chip_counts'],
                    sub_data['places'],
                    sub_data['payouts'],
            ):
                datum = tuple(chip_counts), tuple(places), tuple(payouts)

                if datum not in data_set:
                    data_set.add(datum)
                    data['chip_counts'].append(chip_counts)
                    data['places'].append(places)
                    data['payouts'].append(payouts)

    chip_percentages = list(map(normalize, map(np.array, data['chip_counts'])))
    places = list(map(np.array, data['places']))
    payout_percentages = list(map(normalize, map(np.array, data['payouts'])))
    payout_percentages2 = list(
        starmap(pad_like, zip(payout_percentages, places)),
    )
    targets = list(starmap(np.take, zip(payout_percentages2, places)))
    baselines = list(
        starmap(baseline, zip(chip_percentages, payout_percentages)),
    )
    data = {
        'chip_percentages': np.array(chip_percentages, dtype=np.object_),
        'payout_percentages': np.array(payout_percentages, dtype=np.object_),
        'targets': np.array(targets, dtype=np.object_),
        'baselines': np.array(baselines, dtype=np.object_),
    }

    np.savez(stdout.buffer, **data)


if __name__ == '__main__':
    main()
