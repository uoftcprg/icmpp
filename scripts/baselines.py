from json import dump
from sys import stdin, stdout

from sklearn.metrics import root_mean_squared_error
import numpy as np

from icmpp import cube, exp2_m1, eye, full, icm, log2_1p, weigh_output


def main():
    data = np.load(stdin.buffer)
    inputs = data['inputs']
    outputs = data['outputs']
    weights = data['weights']
    icm_ = data['icm']
    targets = weigh_output(outputs, weights)
    baselines = {
        'FULL': full(inputs),
        'EYE': eye(inputs),
        'ICM': icm_,
        'ICM (EXP2-M1)': icm(inputs, progress=True, weight=exp2_m1),
        'ICM (LOG2-1P)': icm(inputs, progress=True, weight=log2_1p),
        'ICM (SQRT)': icm(inputs, progress=True, weight=np.sqrt),
        'ICM (CBRT)': icm(inputs, progress=True, weight=np.cbrt),
        'ICM (SQUARE)': icm(inputs, progress=True, weight=np.square),
        'ICM (CUBE)': icm(inputs, progress=True, weight=cube),
    }
    losses = {}

    for key, value in baselines.items():
        value = weigh_output(value, weights)
        loss = root_mean_squared_error(targets, value)
        losses[key] = loss

    dump(losses, stdout)


if __name__ == '__main__':
    main()
