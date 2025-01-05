from random import seed
from sys import argv, stdin, stdout

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

from icmpp import import_string, weigh_output

seed(0)
np.random.seed(0)

MODEL_STRING = argv[1]
MODEL = import_string(MODEL_STRING)()
K_FOLD = KFold(shuffle=True, random_state=0)


def sub_main(
        training_inputs,
        training_targets,
        validation_inputs,
        validation_targets,
):
    MODEL.fit(training_inputs, training_targets)

    validation_outputs = MODEL.predict(validation_inputs)
    error = root_mean_squared_error(validation_targets, validation_outputs)

    return error


def main():
    data = np.load(stdin.buffer)
    outputs = data['outputs']
    weights = data['weights']
    icm = data['icm']
    inputs = weigh_output(icm, weights)
    targets = weigh_output(outputs, weights)
    errors = []

    for training_indices, validation_indices in tqdm(
            K_FOLD.split(inputs),
            total=K_FOLD.get_n_splits(inputs),
    ):
        error = sub_main(
            inputs[training_indices],
            targets[training_indices],
            inputs[validation_indices],
            targets[validation_indices],
        )

        errors.append(error)

    errors = np.array(errors)
    data = {'errors': errors}

    np.savez(stdout.buffer, **data)


if __name__ == '__main__':
    main()
