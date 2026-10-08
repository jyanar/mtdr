"""The N-version oracle for the marginal likelihood.

`reference_mmle` is ported from the MATLAB reference only; `mmle_fixture`
loads `tests/fixtures/mmle.mat` into the port's layouts. `tests/` is on
`sys.path` (`tests/conftest.py`), so the tests import these as
`nversion.reference_mmle` and `nversion.mmle_fixture`.
"""
