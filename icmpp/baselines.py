from functools import cache, partial
from itertools import permutations

from scipy.special import logsumexp
import numpy as np

from icmpp.decorators import asserted, unbatched, unpadded


@asserted
@unbatched
@unpadded
def full(input_):
    full = np.full((input_.size, input_.size), 1 / input_.size)

    return full


@asserted
@unbatched
@unpadded
def eye(input_):
    eye = np.eye(input_.size)[np.flip(input_.argsort())].T

    return eye


@cache
def _permute(count):
    return list(permutations(range(count)))


def _help_icm(weights, indices):
    probability = 1
    weight_sum = weights.sum()

    for index in indices:
        weight = weights[index]
        probability *= weight / weight_sum
        weight_sum -= weight

    return probability


@asserted
@unbatched
@unpadded
def icm(input_, *, weight=None):
    if weight is not None:
        input_ = weight(input_)

    icm = np.zeros((input_.size, input_.size))
    indices = _permute(input_.size)
    probabilities = list(map(partial(_help_icm, input_), indices))
    columns = np.arange(input_.size)

    for rows, probability in zip(indices, probabilities):
        icm[rows, columns] += probability

    return icm


def _help_log_icm(log_weights, indices):
    log_probability = 0
    log_weights = log_weights.copy()

    for index in indices:
        log_probability += log_weights[index] - logsumexp(log_weights)
        log_weights[index] = -np.inf

    return log_probability


@asserted
@unbatched
def log_icm(input_, *, weight=None):
    if weight is not None:
        input_ = weight(input_)

    log_icm = np.full((input_.size, input_.size), -np.inf)
    indices = _permute(input_.size)
    log_probabilities = list(map(partial(_help_log_icm, input_), indices))
    columns = np.arange(input_.size)

    for rows, log_probability in zip(indices, log_probabilities):
        log_icm[rows, columns] = np.logaddexp(
            log_icm[rows, columns],
            log_probability,
        )

    return log_icm


@asserted
@unbatched
@unpadded
def mcicm(input_, *, mc_sample_count, weight=None):
    if weight is not None:
        input_ = weight(input_)

    icm = np.zeros((input_.size, input_.size))
    probabilities = input_ / input_.sum()
    columns = np.arange(input_.size)

    for _ in range(mc_sample_count):
        rows = np.random.choice(input_.size, input_.size, False, probabilities)
        icm[rows, columns] += 1

    icm /= mc_sample_count

    return icm
