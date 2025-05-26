from sys import stdin, stdout

from tqdm import trange
import numpy as np
import pandas as pd

from icmpp import icm, mcicm

ICM_COUNT_LIMIT = 10
MCICM_STDERR_TOLERANCE = 0.001
MCICM_MIN_COUNT = 100
MCICM_MAX_COUNT = 10000


def main():
    data = np.load(stdin.buffer, allow_pickle=True)
    chip_percentages = data['chip_percentages']
    payout_percentages = data['payout_percentages']
    targets = data['targets']
    baselines = data['baselines']
    data = []

    for i in trange(chip_percentages.size):
        assert (
            chip_percentages[i].ndim
            == payout_percentages[i].ndim
            == targets[i].ndim
            == baselines[i].ndim
            == 1
        )
        assert (
            chip_percentages[i].size
            == targets[i].size
            == baselines[i].size
            >= payout_percentages[i].size
        )
        assert (chip_percentages[i] > 0).all()
        assert (payout_percentages[i] > 0).all()
        assert (targets[i] >= 0).all()
        assert (baselines[i] >= 0).all()
        assert np.isclose(chip_percentages[i].sum(), 1)
        assert np.isclose(payout_percentages[i].sum(), 1)
        assert np.isclose(targets[i].sum(), 1)
        assert np.isclose(baselines[i].sum(), 1)

        count = chip_percentages[i].size

        assert 2 <= count

        if count <= ICM_COUNT_LIMIT:
            icms = icm(chip_percentages[i], payout_percentages[i])
        else:
            icms, *_ = mcicm(
                chip_percentages[i],
                payout_percentages[i],
                MCICM_STDERR_TOLERANCE,
                MCICM_MIN_COUNT,
                MCICM_MAX_COUNT,
            )

        assert chip_percentages[i].shape == icms.shape
        assert (icms > 0).all()
        assert np.isclose(icms.sum(), 1)

        for j in trange(count, leave=False):
            datum_1 = {
                'end_of_day': i,
                'count': count,
                'player': j,
                'chip_percentage': chip_percentages[i][j],
                'target': targets[i][j],
                'algorithm': 'BASELINE',
                'output': baselines[i][j],
            }
            datum_2 = {
                'end_of_day': i,
                'count': count,
                'player': j,
                'chip_percentage': chip_percentages[i][j],
                'target': targets[i][j],
                'algorithm': 'ICM',
                'output': icms[j],
            }

            data.append(datum_1)
            data.append(datum_2)

    df = pd.DataFrame(data)

    df.to_csv(stdout)


if __name__ == '__main__':
    main()
