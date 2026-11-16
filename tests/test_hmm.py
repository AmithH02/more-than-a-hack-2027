"""The HMM decoder (possession/hmm.py) and hmm_viterbi, on synthetic data."""

import itertools

import numpy as np

from possession import hmm
from possession.experiments import hmm_viterbi

from tests.test_players_fill import synthetic_match

RATES = {"switch": 0.005, "to_dead": 0.004, "from_dead": 0.01}


def emissions(states, strength=1.0):
    """(n, 3) log scores that favour `states` (a string of H/A/D) by `strength` per frame."""
    out = np.zeros((len(states), 3))
    for t, s in enumerate(states):
        out[t, hmm.STATES.index(s)] = strength
    return out


def letters(path):
    return "".join(hmm.STATES[i] for i in path)


def test_a_short_flicker_is_ignored_but_a_long_spell_is_kept():
    truth = "H" * 100 + "A" * 3 + "H" * 100 + "A" * 100
    path = hmm.viterbi(emissions(truth), hmm.transitions(RATES))
    assert letters(path) == "H" * 203 + "A" * 100


def test_a_run_of_dead_ball_evidence_becomes_one_stoppage():
    truth = "H" * 80 + "D" * 60 + "A" * 80
    noisy = emissions(truth)
    noisy[100:103] = emissions("HHH")          # three stray frames inside the stoppage
    path = letters(hmm.viterbi(noisy, hmm.transitions(RATES)))
    assert path == truth


def test_viterbi_matches_brute_force_on_tiny_inputs():
    rng = np.random.default_rng(0)
    T = np.log(rng.dirichlet(np.ones(3), size=3))
    for _ in range(20):
        E = rng.normal(size=(5, 3))
        best = max(itertools.product(range(3), repeat=5),
                   key=lambda p: E[0, p[0]] + sum(T[p[t - 1], p[t]] + E[t, p[t]]
                                                  for t in range(1, 5)))
        assert tuple(hmm.viterbi(E, T)) == best


def test_decode_never_crosses_a_block_and_fills_outside():
    truth = "H" * 50 + "A" * 50
    labels = hmm.decode(emissions(truth), hmm.transitions(RATES), [(0, 49), (60, 99)])
    assert "".join(labels[:50]) == "H" * 50
    assert "".join(labels[50:60]) == "D" * 10
    assert "".join(labels[60:]) == "A" * 40


def test_stickier_switch_needs_longer_evidence():
    truth = "H" * 100 + "A" * 4 + "H" * 100
    loose = hmm.transitions(RATES, switch=0.3)
    sticky = hmm.transitions(RATES, switch=3.0)
    assert "A" in letters(hmm.viterbi(emissions(truth, 2.0), loose))
    assert "A" not in letters(hmm.viterbi(emissions(truth, 2.0), sticky))


def test_fit_rates_counts_transitions():
    rates = hmm.fit_rates(["HHHHAAAADDDD", "HHHHHHHHHHHH"])
    assert all(0 < v < 1 for v in rates.values())
    assert rates["from_dead"] > rates["switch"]   # 3 of 4 D frames stay D; 1 of 18 alive switches


def test_hmm_viterbi_fits_and_predicts_on_a_synthetic_match(tmp_path, monkeypatch):
    match = synthetic_match(n=600)
    path = tmp_path / "hmm.json"
    hmm_viterbi.fit([match], path=str(path), say=lambda *a: None)
    monkeypatch.setattr(hmm_viterbi, "MODEL_PATH", str(path))
    labels = hmm_viterbi.predict_with(match, w=0.2, switch=1.0, dead=1.0)
    again = hmm_viterbi.predict_with(match, w=0.2, switch=1.0, dead=1.0)
    assert labels.shape == (600,) and set(labels) <= {"H", "A", "D"}
    np.testing.assert_array_equal(labels, again)
    hmm_viterbi._load.cache = None
    hmm_viterbi._STATE_LOGP.clear()
