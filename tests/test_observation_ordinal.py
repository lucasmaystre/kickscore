import numpy as np

from kickscore.observation.ordinal import _mm_probit_win, _mm_probit_tie


MEAN_CAV = 1.23
COV_CAV = 4.56
MARGIN = 0.98

# Ground-truth computed using kickscore at revision `66beee380`.
MM_PROBIT_WIN = (
        -0.35804993126636214, 0.21124433823827732, -0.09135628123504448)
MM_PROBIT_TIE = (
        -1.2606613196803695, -0.20881357057246397, -0.16982734815171024)


def test_mm_probit_win():
    """``_mm_probit_win`` should work as expected."""
    assert np.allclose(_mm_probit_win(MEAN_CAV, COV_CAV), MM_PROBIT_WIN)


def test_mm_probit_tie():
    """``_mm_probit_tie`` should work as expected."""
    assert np.allclose(
            _mm_probit_tie(MEAN_CAV, COV_CAV, MARGIN), MM_PROBIT_TIE)
