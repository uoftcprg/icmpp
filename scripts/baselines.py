from sys import stdin, stdout

from tqdm import trange
import numpy as np
import pandas as pd

from icmpp import DEFAULT_COUNT, eye, weigh_output

BASELINE = eye


def main():
    data = np.load(stdin.buffer)
    inputs = data['inputs']
    outputs = data['outputs']
    weights = data['weights']
    icm = data['icm']
    baseline = BASELINE(inputs)
    targets = weigh_output(outputs, weights)
    icm_outputs = weigh_output(icm, weights)
    baseline_outputs = weigh_output(baseline, weights)
    data = []

    for i in trange(len(inputs)):
        count = np.count_nonzero(inputs[i])

        for j in trange(DEFAULT_COUNT, leave=False):
            if not inputs[i, j]:
                assert not targets[i, j]
                assert not icm_outputs[i, j]
                assert not baseline_outputs[i, j]

                continue

            datum_1 = {
                'event': i,
                'count': count,
                'player': j,
                'input': inputs[i, j],
                'target': targets[i, j],
                'algorithm': 'ICM',
                'output': icm_outputs[i, j],
            }
            datum_2 = {
                'event': i,
                'count': count,
                'player': j,
                'input': inputs[i, j],
                'target': targets[i, j],
                'algorithm': 'BASELINE',
                'output': baseline_outputs[i, j],
            }

            data.append(datum_1)
            data.append(datum_2)

    df = pd.DataFrame(data)

    df.to_csv(stdout)


if __name__ == '__main__':
    main()
