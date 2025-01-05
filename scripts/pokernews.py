from functools import partial
from json import dump
from sys import argv, stdout

from tqdm import tqdm
import pandas as pd

from icmpp import DEFAULT_COUNT

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


def get_weight(payouts, event_href, count):
    weight = []

    for i in range(count):
        weight.append(payouts[event_href].get(i, 0))

    return weight


def help_get_data(ranks, payouts, df):
    assert df['event_href'].nunique() == 1

    event_href, *_ = df['event_href']
    chips = list(map(parse_int, df['chips']))

    assert chips == sorted(chips, reverse=True), df

    if 0 in chips:
        input_ = chips[:chips.index(0)]
    else:
        input_ = chips

    players = df['player'][:len(input_)]
    label = list(map(ranks.get(event_href, {}).get, players))

    assert len(input_) == len(label)

    if (
            set(label) != set(range(len(label)))
            or not 2 <= len(label) <= DEFAULT_COUNT
            or event_href not in payouts
    ):
        datum = None
    else:
        weight = get_weight(payouts, event_href, len(label))

        assert len(label) == len(weight)

        datum = input_, label, weight

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
