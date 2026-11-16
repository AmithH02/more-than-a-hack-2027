"""A small hidden Markov model over possession states, decoded by Viterbi. numpy only.

The states are H, A and D (home in possession, away in possession, dead ball), in that order. Each
frame brings evidence about the state (an emission score per state); moving from one state to
another costs a transition penalty. Viterbi finds the single sequence of states with the best total
score over a whole half, so one odd frame cannot flip possession, but evidence that holds up over
time can. That is what spellF1 rewards: few, well-placed hand-overs (CONTEXT.md).

    T = transitions(rates, switch=2.0, dead=1.0)    # (3, 3) log scores
    states = viterbi(log_emit, T)                     # (n,) int, 0 = H, 1 = A, 2 = D
    labels = decode(log_emit, T, blocks)              # (n,) "H"/"A"/"D", each half on its own

Everything is deterministic: ties go to the lowest state index.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2


def fit_rates(label_arrays):
    """{"switch", "to_dead", "from_dead"}: per-frame transition rates counted in `label_arrays`
    (sequences of "H"/"A"/"D", one per half or match). Laplace-smoothed so none is zero."""
    counts = np.ones((3, 3))
    index = {s: i for i, s in enumerate(STATES)}
    for labels in label_arrays:
        codes = np.array([index[l] for l in labels], dtype=np.int64)
        np.add.at(counts, (codes[:-1], codes[1:]), 1)
    alive = counts[:2].sum()
    return {
        "switch": float((counts[H, A] + counts[A, H]) / alive),
        "to_dead": float((counts[H, D] + counts[A, D]) / alive),
        "from_dead": float((counts[D, H] + counts[D, A]) / counts[D].sum()),
    }


def transitions(rates, switch=1.0, dead=1.0):
    """(3, 3) log transition scores from per-frame `rates` (fit_rates). `switch` and `dead`
    multiply the log cost of a hand-over (H<->A) and of entering or leaving D: above 1 makes the
    decoder stickier than the counted rates, below 1 more willing to change."""
    s, td, fd = rates["switch"], rates["to_dead"], rates["from_dead"]
    T = np.empty((3, 3))
    T[H, H] = T[A, A] = np.log(1 - s - td)
    T[H, A] = T[A, H] = switch * np.log(s)
    T[H, D] = T[A, D] = dead * np.log(td)
    T[D, D] = np.log(1 - fd)
    T[D, H] = T[D, A] = dead * np.log(fd / 2)
    return T


def viterbi(log_emit, log_trans, log_start=None):
    """(n,) int: the state sequence with the highest total of emission and transition scores.

    log_emit is (n, k); log_trans is (k, k), from row to column; log_start (k,) defaults to 0.
    """
    log_emit = np.asarray(log_emit, dtype=np.float64)
    n, k = log_emit.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    T = np.asarray(log_trans, dtype=np.float64)
    back = np.empty((n, k), dtype=np.int8)
    score = log_emit[0] + (np.zeros(k) if log_start is None else np.asarray(log_start))
    cols = np.arange(k)
    for t in range(1, n):
        cand = score[:, None] + T                   # cand[i, j]: be in i, then move to j
        best = cand.argmax(axis=0)                  # first maximum: ties to the lowest index
        back[t] = best
        score = cand[best, cols] + log_emit[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(log_emit, log_trans, blocks, outside="D"):
    """(n,) "H"/"A"/"D" labels: Viterbi on each block (first, last inclusive) separately, never
    across halftime; frames outside every block get `outside`."""
    labels = np.full(len(log_emit), outside, dtype="<U1")
    letters = np.array(STATES)
    for first, last in blocks:
        labels[first:last + 1] = letters[viterbi(log_emit[first:last + 1], log_trans)]
    return labels
