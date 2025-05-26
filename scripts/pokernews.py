from functools import partial
from json import dump
from sys import argv, stdout
from warnings import warn

from tqdm import tqdm
import pandas as pd

CHIPS_PATH = argv[1]
PAYOUTS_PATH = argv[2]


def help_get_ranks(df):
    places = list(map(int, df['place']))
    players = df['player']
    ranks = {}

    for place, player in zip(places, players):
        ranks.setdefault(player, place - 1)

    return ranks


def get_ranks(df):
    df = df.loc[df['player'].notnull()]
    ranks = df.groupby('event_href').progress_apply(help_get_ranks)

    return dict(ranks)


def parse_int(raw_int):
    return int(
        (
            raw_int
            .replace(',', '')
            .lstrip('NRTZ$£€₹')
            .rstrip('ABCDFGHKLMNPRSUVWZ')
        ),
    )


def help_get_payouts(df):
    places = list(map(int, df['place']))
    winnings = list(map(parse_int, df['winning']))
    payouts = {}

    for place, winning in zip(places, winnings):
        payouts.setdefault(place - 1, winning)

    return payouts


def get_payouts(df):
    df = df.loc[df['place'].notnull()]
    payouts = df.groupby('event_href').progress_apply(help_get_payouts)

    return dict(payouts)


def get_payout(payouts, event_href, count):
    payout = []

    for i in range(count):
        payout.append(payouts[event_href].get(i, 0))

    if payout != sorted(payout, reverse=True):
        warn(f'Unsorted payouts {payout}')

    while payout and not payout[-1]:
        payout.pop()

    return payout


def help_get_data(ranks, payouts, df):
    assert df['event_href'].nunique() == 1

    event_href, *_ = df['event_href']
    chips = list(map(parse_int, df['chips']))

    assert chips == sorted(chips, reverse=True), df

    if 0 in chips:
        chip_count = chips[:chips.index(0)]
    else:
        chip_count = chips

    players = df['player'][:len(chip_count)]
    place = list(map(ranks.get(event_href, {}).get, players))

    assert len(chip_count) == len(place)

    if (
            set(place) == set(range(len(place)))
            and len(place) >= 2
            and event_href in payouts
            and (payout := get_payout(payouts, event_href, len(place)))
    ):
        datum = chip_count, place, payout
    else:
        datum = None

    return datum


def get_data(ranks, payouts, df):
    df = df.loc[df['player'].notnull()].drop_duplicates()
    data = (
        df
        .groupby(['day_href'])
        .progress_apply(partial(help_get_data, ranks, payouts))
    )

    return list(filter(None, data))


def main():
    tqdm.pandas()

    chips_df = pd.read_csv(CHIPS_PATH, index_col=0, dtype=str)
    payouts_df = pd.read_csv(PAYOUTS_PATH, index_col=0, dtype=str)
    ranks = get_ranks(payouts_df)
    payouts = get_payouts(payouts_df)
    data = get_data(ranks, payouts, chips_df)
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
