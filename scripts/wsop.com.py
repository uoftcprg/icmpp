from functools import partial
from json import dump
from sys import argv, stdout
from warnings import warn

from tqdm import tqdm
import pandas as pd

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


def get_payout(payouts, tid, count):
    payout = []

    for i in range(count):
        payout.append(payouts[tid].get(i, 0))

    if payout != sorted(payout, reverse=True):
        warn(f'Unsorted payouts {payout}')

    while payout and not payout[-1]:
        payout.pop()

    return payout


def help_get_data(ranks, payouts, df):
    assert df['tid'].nunique() == 1

    tid, *_ = df['tid']
    chipstacks = list(map(parse_int, df['chipstack']))

    assert chipstacks == sorted(chipstacks, reverse=True)

    if 0 in chipstacks:
        chip_count = chipstacks[:chipstacks.index(0)]
    else:
        chip_count = chipstacks

    players = list(map(normalize_str, df['player']))[:len(chip_count)]
    place = list(map(ranks.get(tid, {}).get, players))

    assert len(chip_count) == len(place)

    if (
            set(place) == set(range(len(place)))
            and len(place) >= 2
            and tid in payouts
            and (payout := get_payout(payouts, tid, len(place)))
    ):
        datum = chip_count, place, payout
    else:
        datum = None

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
    chip_counts = []
    places = []
    payouts = []

    for chip_count, place, payout in data:
        chip_counts.append(chip_count)
        places.append(place)
        payouts.append(payout)

    data = {'chip_counts': chip_counts, 'places': places, 'payouts': payouts}

    dump(data, stdout)


if __name__ == '__main__':
    main()
