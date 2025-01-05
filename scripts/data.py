from collections import defaultdict
from json import load
from sys import argv, stdout

import numpy as np

from icmpp import (
    assert_input_and_label,
    assert_input_and_output,
    DEFAULT_COUNT,
    icm,
    normalize_input,
    pad_input,
    pad_label,
    pad_output,
)

DATA_PATHS = argv[1:]


def preprocess_input(input_):
    input_ = np.array(input_)
    input_ = pad_input(input_, DEFAULT_COUNT)
    input_ = normalize_input(input_)

    return input_


def preprocess_label(label):
    label = np.array(label)
    label = pad_label(label, DEFAULT_COUNT)

    return label


def preprocess_output(label):
    label = np.array(label)
    output = np.eye(label.size)[label]
    output = pad_output(output, DEFAULT_COUNT)

    return output


def preprocess_weight(weight):
    return preprocess_input(weight)


def main():
    input_label_weights = set()
    data = defaultdict(list)

    for path in DATA_PATHS:
        with open(path) as file:
            sub_data = load(file)

            for input_, label, weight in zip(
                    sub_data['inputs'],
                    sub_data['labels'],
                    sub_data['weights'],
            ):
                input_label_weight = (
                    tuple(input_),
                    tuple(label),
                    tuple(weight),
                )

                if input_label_weight not in input_label_weights:
                    input_label_weights.add(input_label_weight)
                    data['inputs'].append(input_)
                    data['labels'].append(label)
                    data['weights'].append(weight)

    inputs = np.stack(list(map(preprocess_input, data['inputs'])))
    labels = np.stack(list(map(preprocess_label, data['labels'])))
    outputs = np.stack(list(map(preprocess_output, data['labels'])))
    weights = np.stack(list(map(preprocess_weight, data['weights'])))
    icm_ = icm(inputs, progress=True)

    assert_input_and_label(inputs, labels)
    assert_input_and_output(inputs, outputs)

    data = {
        'inputs': inputs,
        'labels': labels,
        'outputs': outputs,
        'weights': weights,
        'icm': icm_,
    }

    np.savez(stdout.buffer, **data)


if __name__ == '__main__':
    main()
