from functools import partial
from json import dump
from sys import argv, stdout

from tqdm import tqdm
import pandas as pd

from icmpp import DEFAULT_COUNT

RESULTS_PATH = argv[1]
CHIPCOUNTS_PATH = argv[2]


def normalize_str(raw_str):
    return ' '.join(raw_str.split()).lower()


def help_get_ranks(df):
    places = list(map(int, df['place']))
    players = list(map(normalize_str, df['player']))
    ranks = {}

    for place, player in zip(places, players):
        ranks.setdefault(player, place - 1)

    return ranks


def get_ranks(df):
    df = df.loc[df['player'].notnull()]
    ranks = df.groupby('tid').progress_apply(help_get_ranks)

    return dict(ranks)


def parse_int(raw_int):
    return int(raw_int.replace(',', '').lstrip('ACDFMRXex$£€¥'))


def help_get_payouts(df):
    places = list(map(int, df['place']))
    awards = list(map(parse_int, df['award']))
    payouts = {}

    for place, award in zip(places, awards):
        payouts.setdefault(place - 1, award)

    return payouts


def get_payouts(df):
    payouts = df.groupby('tid').progress_apply(help_get_payouts)

    return dict(payouts)


def get_weight(payouts, tid, count):
    weight = []

    for i in range(count):
        weight.append(payouts[tid].get(i, 0))

    return weight


def help_get_data(ranks, payouts, df):
    assert df['tid'].nunique() == 1

    tid, *_ = df['tid']
    chipstacks = list(map(parse_int, df['chipstack']))

    assert chipstacks == sorted(chipstacks, reverse=True)

    if 0 in chipstacks:
        input_ = chipstacks[:chipstacks.index(0)]
    else:
        input_ = chipstacks

    players = list(map(normalize_str, df['player']))[:len(input_)]
    label = list(map(ranks.get(tid, {}).get, players))

    assert len(input_) == len(label)

    if (
            set(label) != set(range(len(label)))
            or not 2 <= len(label) <= DEFAULT_COUNT
            or tid not in payouts
    ):
        datum = None
    else:
        weight = get_weight(payouts, tid, len(label))

        assert len(label) == len(weight)

        datum = input_, label, weight

    return datum


def get_data(ranks, payouts, df):
    df = df.loc[df[['player', 'chipstack']].notnull().all(axis=1)]
    data = (
        df
        .groupby(['tid', 'dayof', 'day'])
        .progress_apply(partial(help_get_data, ranks, payouts))
    )

    return list(filter(None, data))


def main():
    tqdm.pandas()

    results_df = pd.read_csv(RESULTS_PATH, index_col=0, dtype=str)
    chipcounts_df = pd.read_csv(CHIPCOUNTS_PATH, index_col=0, dtype=str)
    ranks = get_ranks(results_df)
    payouts = get_payouts(results_df)
    data = get_data(ranks, payouts, chipcounts_df)
    inputs = []
    labels = []
    weights = []

    for input_, label, weight in data:
        inputs.append(input_)
        labels.append(label)
        weights.append(weight)

    data = {'inputs': inputs, 'labels': labels, 'weights': weights}

    dump(data, stdout)


if __name__ == '__main__':
    main()
