from itertools import count as _count, permutations

import numpy as np

from icmpp.utilities import pad_like


def baseline(chip_percentages, payouts):
    payouts = pad_like(payouts, chip_percentages)

    return payouts[np.flip(chip_percentages.argsort())]


def icm(chip_percentages, payouts):
    if payouts.size > chip_percentages.size:
        payouts = payouts[:chip_percentages.size]

    estimates = np.zeros_like(chip_percentages)

    for indices in map(
            np.array,
            permutations(range(chip_percentages.size), payouts.size),
    ):
        probability = 1
        denominator = 1

        for i in indices:
            chip_percentage = chip_percentages[i]
            probability *= chip_percentage / denominator
            denominator -= chip_percentage

        estimates[indices] += payouts * probability

    return estimates


def mcicm(chip_percentages, payouts, stderr_tolerance, min_count, max_count):
    if payouts.size > chip_percentages.size:
        payouts = payouts[:chip_percentages.size]

    mean = np.zeros_like(chip_percentages)
    delta_square_sum = np.zeros_like(chip_percentages)
    stderr = np.full_like(chip_percentages, np.inf)
    count = 0

    for count in _count(1):
        indices = np.random.choice(
            chip_percentages.size,
            payouts.size,
            False,
            chip_percentages,
        )
        sample = np.zeros_like(chip_percentages)
        sample[indices] = payouts

        delta = sample - mean
        mean += delta / count
        delta2 = sample - mean
        delta_square_sum += delta * delta2

        if count > 1:
            variance = delta_square_sum / (count - 1)
            stderr = np.sqrt(variance / count)

            if (stderr < stderr_tolerance).all() and count > min_count:
                break

        if count == max_count:
            break

    return mean, stderr, count


def ebcm(chip_percentages, payouts):
    if payouts.size > chip_percentages.size:
        payouts = payouts[:chip_percentages.size]

    inverse_chip_percentages = np.reciprocal(chip_percentages)
    inverse_chip_percentages /= inverse_chip_percentages.sum()
    reverse_payouts = np.flip(payouts)

    estimates = np.zeros_like(chip_percentages)

    for reverse_indices in map(
            np.array,
            permutations(range(chip_percentages.size), payouts.size),
    ):
        probability = 1
        denominator = 1

        for i in reverse_indices:
            inverse_chip_percentage = inverse_chip_percentages[i]
            probability *= inverse_chip_percentage / denominator
            denominator -= inverse_chip_percentage

        estimates[reverse_indices] += reverse_payouts * probability

    return estimates


def mcebcm(chip_percentages, payouts, stderr_tolerance, min_count, max_count):
    if payouts.size > chip_percentages.size:
        payouts = payouts[:chip_percentages.size]

    inverse_chip_percentages = np.reciprocal(chip_percentages)
    inverse_chip_percentages /= inverse_chip_percentages.sum()
    reverse_payouts = np.flip(payouts)

    mean = np.zeros_like(chip_percentages)
    delta_square_sum = np.zeros_like(chip_percentages)
    stderr = np.full_like(chip_percentages, np.inf)
    count = 0

    for count in _count(1):
        reverse_indices = np.random.choice(
            chip_percentages.size,
            payouts.size,
            False,
            inverse_chip_percentages,
        )
        sample = np.zeros_like(chip_percentages)
        sample[reverse_indices] = reverse_payouts

        delta = sample - mean
        mean += delta / count
        delta2 = sample - mean
        delta_square_sum += delta * delta2

        if count > 1:
            variance = delta_square_sum / (count - 1)
            stderr = np.sqrt(variance / count)

            if (stderr < stderr_tolerance).all() and count > min_count:
                break

        if count == max_count:
            break

    return mean, stderr, count
