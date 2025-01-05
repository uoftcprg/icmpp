from importlib import import_module

from sklearn.preprocessing import normalize
import numpy as np

IGNORE_INDEX = -100
DEFAULT_COUNT = 9


def normalize_input(input_):
    if input_.ndim == 1:
        input_ = normalize(input_.reshape(1, -1), 'l1').flatten()
    else:
        input_ = normalize(input_, 'l1')

    return input_


def pad_input(input_, count=DEFAULT_COUNT):
    pad_count = count - input_.size

    if pad_count:
        input_ = np.pad(input_, (0, pad_count))

    return input_


def pad_label(label, count=DEFAULT_COUNT):
    pad_count = count - label.size

    if pad_count:
        label = np.pad(label, (0, pad_count), constant_values=IGNORE_INDEX)

    return label


def pad_output(output, count=DEFAULT_COUNT):
    pad_count = count - output.shape[0]

    if pad_count:
        output = np.pad(output, ((0, pad_count), (0, pad_count)))
        output[-pad_count:, -pad_count:] = 1 / pad_count

    return output


def assert_input(input_):
    assert 1 <= input_.ndim <= 2
    assert (0 <= input_).all()


def assert_label(label):
    assert 1 <= label.ndim <= 2


def assert_output(output):
    assert 2 <= output.ndim <= 3
    assert output.shape[-1] == output.shape[-2]
    assert ((0 <= output) & (output <= 1)).all()
    assert np.allclose(output.sum(-1), 1)
    assert np.allclose(output.sum(-2), 1)


def assert_input_and_label(input_, label):
    assert input_.shape == label.shape
    assert np.array_equal(input_ == 0, label == IGNORE_INDEX)

    assert_input(input_)
    assert_label(label)


def assert_input_and_output(input_, output):
    assert input_.ndim + 1 == output.ndim
    assert input_.shape[0] == output.shape[0]
    assert (
        input_.ndim == 1
        or (input_.shape[1] == output.shape[1] == output.shape[2])
    )

    assert_input(input_)
    assert_output(output)


def exp2_m1(x):
    return np.exp2(x) - 1


def log2_1p(x):
    return np.log2(1 + x)


def cube(x):
    return 3 ** x


def weigh_output(output, weight):
    return (output @ weight[:, :, np.newaxis]).reshape(-1, weight.shape[-1])


def import_string(string):
    module_name, attribute_name = string.rsplit('.', 1)
    module = import_module(module_name)
    attribute = getattr(module, attribute_name)

    return attribute
