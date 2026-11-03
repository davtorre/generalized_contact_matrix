"""Tests for Counts and for ContactMatrix.from_numpy / to_numpy round trips.

Run from the project root, either with pytest (if installed) or directly:

    python -m pytest tests
    python tests/test_matrix.py

The package must be installed in the environment first: pip install -e .  (see README)
"""
from pathlib import Path

import numpy as np
import polars as pl

from gcm import ContactMatrix, Counts, Domain, Element

ROOT = Path(__file__).resolve().parent.parent

POLYMOD_CSV = ROOT / "data" / "processed" / "polymod_contact_matrix.csv"


def raises(error, f):
    """True if calling f raises `error` (keeps the tests free of a pytest dependency)."""
    try:
        f()
    except error:
        return True
    return False


def test_round_trip_small():
    age = Domain("age", [0, 5, 10])
    R = [[12, 3, 1],
         [4, 9, 2],
         [1, 2, 6]]
    cm = ContactMatrix.from_numpy(R, age, age, N_part=[50, 40, 30], N_cont=[7, 8, 9])

    R2, rows, cols = cm.to_numpy()
    assert np.array_equal(R2, np.array(R))
    assert R2.dtype.kind == "i"
    assert isinstance(rows, Domain) and rows.elements == cols.elements == [(0,), (5,), (10,)]
    assert rows[1] == (5,) and rows[-1] == (10,)
    assert dict(cm.N_part) == {(0,): 50, (5,): 40, (10,): 30}
    assert dict(cm.N_cont) == {(0,): 7, (5,): 8, (10,): 9}
    assert cm.contacts[(5,)][(0,)] == 4


def test_round_trip_rectangular_product_domain():
    part = Domain("age", [0, 5, 10]) * Domain("gender", ["F", "M"])     # 6 rows
    cont = Domain("age", [0, 5, 10])                                     # 3 columns
    R = np.random.default_rng(0).integers(0, 100, size=(6, 3))
    cm = ContactMatrix.from_numpy(R, part, cont, N_part=np.arange(1, 7))

    R2, rows, cols = cm.to_numpy()
    assert np.array_equal(R2, R)
    assert rows == part and cols == cont
    assert cm.N_part[(5, "M")] == 4 and isinstance(cm.N_part[(5, "M")], int)
    assert all(n is None for _, n in cm.N_cont)                     # not given: unknown


def test_round_trip_floats_are_kept():
    age = Domain("age", [0, 5])
    R = np.array([[1.5, 0.25], [2.0, 3.75]])
    R2, _, _ = ContactMatrix.from_numpy(R, age, age).to_numpy()
    assert R2.dtype.kind == "f"
    assert np.array_equal(R2, R)


def test_domains_are_copied():
    age = Domain("age", [0, 5])
    cm = ContactMatrix.from_numpy([[1, 2], [3, 4]], age, age)
    cm.domain_part.add(10)
    assert len(age) == 2


def test_invalid_input():
    age = Domain("age", [0, 5])
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([[1, 2, 3]], age, age))            # shape
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([1, 2], age, age))                 # 1D
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([[1, -2], [3, 4]], age, age))      # negative
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([[1, np.nan], [3, 4]], age, age))  # NaN
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([["a", "b"], ["c", "d"]], age, age))
    assert raises(ValueError, lambda: ContactMatrix.from_numpy([[1, 2], [3, 4]], age, age, N_part=[1]))
    assert raises(TypeError, lambda: ContactMatrix.from_numpy([[1, 2], [3, 4]], ["age"], age))


def test_round_trip_polymod():
    if not POLYMOD_CSV.exists():
        print("skipped: run scripts/make_dataset.py first (see README)")
        return
    df = pl.read_csv(POLYMOD_CSV)
    cm = ContactMatrix().from_dataframe(df)
    R, rows, cols = cm.to_numpy()

    cm2 = ContactMatrix.from_numpy(R, cm.domain_part, cm.domain_cont,
                                   N_part=cm.N_part, N_cont=cm.N_cont)                 # Counts, looked up by group
    R2, rows2, cols2 = cm2.to_numpy()
    assert R2.shape == (240, 30) and R2.sum() == 95784
    assert np.array_equal(R2, R) and rows2 == rows and cols2 == cols
    # and all the way back to the dataset
    assert cm2.to_dataframe().equals(df)


# ---- domains returned by to_numpy --------------------------------------------------------

def small_matrix():
    """Rows age x gender (4 groups), columns age (2 groups); R[i, j] = 10 * i + j."""
    part = Domain("age", [0, 5]) * Domain("gender", ["F", "M"])
    cont = Domain("age", [0, 5])
    R = np.array([[0, 1], [10, 11], [20, 21], [30, 31]])
    return ContactMatrix.from_numpy(R, part, cont), R


def test_returned_domains_are_copies():
    cm, _ = small_matrix()
    _, dp, _ = cm.to_numpy()
    dp.add((10, "F"))
    assert len(cm.domain_part) == 4


def test_from_numpy_of_to_numpy():
    cm, R = small_matrix()
    R2, dp, dc = ContactMatrix.from_numpy(*cm.to_numpy()).to_numpy()
    assert np.array_equal(R2, R) and dp == cm.domain_part and dc == cm.domain_cont

# ---- Counts ----------------------------------------------------------------------------------

def groups():
    return Domain("age", ["0-4", "5-9"]) * Domain("eth", ["hispanic", "white"]) * Domain("gender", ["F", "M"])


def test_counts_get_set_add():
    N = Counts(groups(), range(8))
    feat = ("5-9", "hispanic", "F")
    assert N[feat] == 4
    N[feat] = 40
    N.add(feat, 2)
    assert N[feat] == 42 and N.get(feat) == 42
    assert N.get(("10-14", "white", "M")) is None and N.get(("10-14", "white", "M"), 0) == 0
    assert raises(KeyError, lambda: N[("10-14", "white", "M")])


def test_counts_unknown_is_not_zero():
    N = Counts(Domain("age", [0, 5, 10]), [3, None, float("nan")])
    assert N[5] is None and N[10] is None and N.get(5, 0) == 0
    N.add(5, 2)                                                  # unknown + 2 -> 2
    values, _ = N.to_numpy()
    assert values.dtype.kind == "f" and values[1] == 2 and np.isnan(values[2])
    assert Counts(Domain("age", [0, 5])).get(0) is None          # no counts given: all unknown


def test_counts_new_groups_and_fixed_domains():
    N = Counts(Domain("age", [0, 5]), [1, 2])
    N[10] = 7                                                    # new group: added to N's domain
    assert len(N) == 3 and N.domain.elements[-1] == (10,)
    fixed = Counts(Domain("age", [0, 5], fixed=True), [1, 2])
    assert raises(ValueError, lambda: fixed.__setitem__(10, 7))


def test_counts_copies_its_domain_and_round_trips():
    d = Domain("age", [0, 5])
    N = Counts(d, np.array([4, 6]))
    N[10] = 1
    assert len(d) == 2                                           # the caller's domain is untouched
    values, domain = N.to_numpy()
    assert values.dtype.kind == "i" and values.tolist() == [4, 6, 1]
    assert Counts.from_numpy(values, domain) == N


def test_counts_invalid():
    d = Domain("age", [0, 5])
    assert raises(ValueError, lambda: Counts(d, [1]))            # wrong length
    assert raises(ValueError, lambda: Counts(d, [1, -2]))        # negative
    assert raises(ValueError, lambda: Counts(d, [1, "a"]))       # not a number
    assert raises(TypeError, lambda: Counts(["age"], [1]))


def test_matrix_counts_follow_the_domain():
    cm = ContactMatrix([Element({"age": 0}, {"age": 5}, 10, 3, 4)])
    cm.add_element(Element({"age": 5, "eth": "w"}, {"age": 5, "sex": "f"}, 1, 2, 3))
    # new features: the old groups get None, and N_part / N_cont follow the shared domains
    assert dict(cm.N_part) == {(0, None): 3, (5, "w"): 2}
    assert dict(cm.N_cont) == {(5, None): 4, (5, "f"): 3}
    assert cm.N_part.domain is cm.domain_part


def test_from_numpy_with_counts_in_any_order():
    age = Domain("age", [0, 5, 10])
    N = Counts(Domain("age", [10, 0, 5]), [30, 50, 40])          # same groups, different order
    cm = ContactMatrix.from_numpy(np.eye(3, dtype=int), age, age, N_part=N)
    assert [n for _, n in cm.N_part] == [50, 40, 30]
    assert raises(ValueError, lambda: ContactMatrix.from_numpy(np.eye(3, dtype=int), age, age,
                                                               N_part=Counts(Domain("age", [0]), [1])))


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
