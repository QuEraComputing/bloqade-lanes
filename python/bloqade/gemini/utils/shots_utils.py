import numpy as np


# helper functions to analyze statistical distribution of logical measurements
def get_histogram(shots: np.ndarray) -> np.ndarray:
    """
    Takes in a 2D numpy array of shots of shape (n_shots, n_outcomes), and returns an array of shape (n_outcomes) where
    array[i] is the frequency of occurence of int `i` expressed in binary in the outcome. The output represents a histogram
    of the input.

    Inputs:
        shots (np.ndarray): 2D array of shape (n_shots, n_outcomes)

    Outputs:
        A numpy array of shape (n_outcomes) where the i-th element correpsonds to the number of times that `i`
        expressed in binary appeared in the input `shots` array.
    """
    n = shots.shape[1]
    weights = 1 << np.arange(n - 1, -1, -1)
    indices = shots.astype(np.int64) @ weights
    return np.bincount(indices, minlength=1 << n)
