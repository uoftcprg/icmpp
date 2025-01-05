from functools import partial, wraps

from tqdm import tqdm
import numpy as np

from icmpp.utilities import assert_input_and_output, pad_output


def asserted(function):

    @wraps(function)
    def wrapper(input_, **kwargs):
        output = function(input_, **kwargs)

        assert_input_and_output(input_, output)

        return output

    return wrapper


def unbatched(function):

    @wraps(function)
    def wrapper(input_, *, progress=False, **kwargs):
        if input_.ndim == 1:
            output = function(input_, **kwargs)
        else:
            if progress:
                input_ = tqdm(input_, leave=False)

            output = np.stack(list(map(partial(function, **kwargs), input_)))

        return output

    return wrapper


def unpadded(function):

    @wraps(function)
    def wrapper(input_, **kwargs):
        if input_.ndim != 1:
            raise ValueError('input is batched')

        count = input_.size
        input_ = input_[~np.isclose(input_, 0)]
        output = function(input_, **kwargs)
        output = pad_output(output, count)

        return output

    return wrapper
