import numpy as np
import pytest
import logging
import scipy.sparse
from thecov import geometry, base, utils
from thecov import math as thecov_math
from mockfactory.make_survey import RandomBoxCatalog
import os
import glob
from mpi4py import MPI

def get_cache_dir():
	return os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache/")

def create_basic_randoms(num_tracers):

	nbar = np.random.rand(num_tracers) * 1e-5
	boxsize = 1000.0

	randoms = []
	for t in range(4):
		if t+1 <= num_tracers:
			randoms.append(RandomBoxCatalog(nbar=nbar[t], boxsize=boxsize))
			randoms[t]["POSITION"] = randoms[t]["Position"]
		else:
			randoms.append(None)
	return randoms

def make_surveywindow_stub():
    # Create an instance of SurveyWindow without running __init__ to avoid heavy setup
    w = geometry.SurveyWindow.__new__(geometry.SurveyWindow)
    w.logger = logging.getLogger('SurveyWindow')
    return w

def test_rebin_parameters_sets_attributes_and_returns_expected_values():

    w = make_surveywindow_stub()
    w.boxsize = 100.0
    w.nmesh = 64

    dk = 0.1
    kmax = 0.02
    trim_to_nmesh, rebin_factor = w._rebin_parameters(dk=dk, kmax=kmax)

    # Recompute expected values the same way as the implementation
    target_boxsize = 2 * np.pi / dk
    target_nmesh = int(np.ceil(target_boxsize * kmax / np.pi))
    expected_trim = int(np.ceil(target_boxsize / w.boxsize * w.nmesh))
    if expected_trim % 2 != 0:
        expected_trim += 1
    expected_rebin = expected_trim // target_nmesh

    assert trim_to_nmesh == expected_trim
    assert rebin_factor == expected_rebin
    assert hasattr(w, 'kboxsize') and hasattr(w, 'knmesh')
    assert np.isclose(w.kboxsize, trim_to_nmesh / w.nmesh * w.boxsize)
    assert w.knmesh == trim_to_nmesh // rebin_factor


def test_rebin_parameters_raises_when_rebin_factor_zero():
    w = make_surveywindow_stub()
    w.boxsize = 100.0
    w.nmesh = 8

    dk = 0.1
    # Choose kmax large so target_nmesh > trim_to_nmesh and rebin_factor == 0
    kmax = 10.0

    with pytest.raises(ZeroDivisionError):
        w._rebin_parameters(dk=dk, kmax=kmax)


def test_ikgrid_returns_wrapped_indices():

    w = make_surveywindow_stub()
    w.nmesh = 8

    ikgrid = w.ikgrid
    assert len(ikgrid) == 3
    expected = np.arange(8)
    expected[expected >= 8 // 2] -= 8

    for axis in ikgrid:
        assert np.array_equal(axis, expected)


def test_knyquist_and_kfun_use_knmesh_when_present():
    w = make_surveywindow_stub()
    # case using knmesh/kboxsize
    w.knmesh = 10
    w.kboxsize = 5.0
    assert np.isclose(w.knyquist, np.pi * w.knmesh / w.kboxsize)
    assert np.isclose(w.kfun, 2 * np.pi / w.kboxsize)

    # case falling back to nmesh/boxsize
    w2 = make_surveywindow_stub()
    w2.nmesh = 16
    w2.boxsize = 8.0
    assert np.isclose(w2.knyquist, np.pi * w2.nmesh / w2.boxsize)
    assert np.isclose(w2.kfun, 2 * np.pi / w2.boxsize)

# ---------------------------------------------------------------------------
# Gaunt coefficient values
#
# G(l1,l2,l3,m1,m2,m3) below denotes the real Gaunt coefficient
# \int Y_l1m1 Y_l2m2 Y_l3m3 dOmega for *real* spherical harmonics, i.e.
# sympy.physics.wigner.real_gaunt(l1,l2,l3,m1,m2,m3).
# ---------------------------------------------------------------------------

GAUNT_ELLMAX = 2  # keeps every array <= 10^4 x 10^2; the whole set builds in <1s

G_000 = 0.28209479177387814  # G(0,0,0,0,0,0) = G(2,0,2,0,0,0)
                             # = G(2,2,0,0,0,0) = G(0,2,2,0,0,0) = 1/(2*sqrt(pi))
G_222 = 0.18022375157286857  # G(2,2,2,0,0,0) = sqrt(5/pi)/7

# Coefficient shared by the second cosmic variance and fourth mixed terms at
# l1=l2=l3=2, la=2, all m=0. Both sum over an intermediate (lc, mc):
#   lc=0 (mc=0)  ->  G(2,2,0,0,0,0) * G(0,2,2,0,0,0)
#   lc=2         ->  only mc=0 survives m-selection: G(2,2,2,0,0,0)^2
LC_SUM = G_000 * G_000 + G_222 * G_222

@pytest.fixture(scope="module")
def gaunt_coefficients(tmp_path_factory):
    """All Gaunt coefficient arrays at pk_ellmax = mask_ellmax = GAUNT_ELLMAX.

    Uses a throwaway cache directory so the arrays are always recomputed (never
    served from a stale cache) and so this never races with the MPI test above,
    which clears the shared cache directory.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Every rank must agree on the cache path, otherwise only rank 0 writes the
    # file and the others fail to load it.
    cache_dir = str(tmp_path_factory.mktemp("gaunt_values")) if rank == 0 else None
    cache_dir = comm.bcast(cache_dir, root=0)

    kwargs = dict(mask_ellmax=GAUNT_ELLMAX, pk_ellmax=GAUNT_ELLMAX,
                  cache_dir=cache_dir, rank=rank, comm=comm)
    sg = geometry.SurveyGeometry
    return {
        "first_cosmic_variance": sg.get_first_cosmic_variance_gaunt_coefficients(**kwargs),
        "second_cosmic_variance": sg.get_second_cosmic_variance_gaunt_coefficients(**kwargs),
        "mixed_first": sg.get_mixed_gaunt_coefficients(term="first", **kwargs),
        "mixed_fourth": sg.get_mixed_gaunt_coefficients(term="fourth", **kwargs),
        "shotnoise": sg.get_shotnoise_gaunt_coefficients(**kwargs),
    }

# (array, index, expected value, description). Index layouts, as documented in
# geometry.py, are (shape_out..., shape_in...) with ell stored as l//2 and m as
# m+l:
#   *_cosmic_variance : (l1,l2,l3,l4, m1,m2,m3,m4) + (la,lb, ma,mb)
#   mixed_*           : (l1,l2,l3, m1,m2,m3)       + (la,lb, ma,mb)
#   shotnoise         : (l1,l2, m1,m2)             + (la,lb, ma,mb)
GAUNT_VALUE_CASES = [
    # -- first cosmic variance: G(l1,l4,la,m1,m4,ma) * G(l2,l3,lb,m2,m3,mb) ----
    ("first_cosmic_variance", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), G_000 * G_000,
     "all ell zero -> G(0,0,0)^2 = 1/(4 pi)"),
    ("first_cosmic_variance", (1, 0, 0, 1, 2, 0, 0, 2, 1, 0, 2, 0), G_222 * G_000,
     "l1=l4=2 couple through la=2; l2=l3=lb=0"),
    # The next two are mirror images: the (l1,l4) pair must couple through la and
    # the (l2,l3) pair through lb. Swapping the two halves would break them.
    ("first_cosmic_variance", (1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0), G_000 * G_000,
     "l1=2 alone -> la=2, lb=0"),
    ("first_cosmic_variance", (0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2), G_000 * G_000,
     "l2=2 alone -> la=0, lb=2"),

    # -- second cosmic variance: sums over the intermediate (lc, mc) -----------
    ("second_cosmic_variance", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), G_000 * G_000,
     "all ell zero -> only lc=0 contributes"),
    ("second_cosmic_variance", (1, 1, 1, 0, 2, 2, 2, 0, 1, 0, 2, 0), LC_SUM,
     "lc=0 and lc=2 both land in la=2 and must be summed, not overwritten"),

    # -- mixed terms -----------------------------------------------------------
    ("mixed_first", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), G_000,
     "first mixed term is a single Gaunt G(l2,l3,lb,m2,m3,mb)"),
    ("mixed_first", (1, 1, 1, 2, 2, 2, 1, 1, 2, 2), G_222,
     "la,ma track (l1,m1); the Gaunt only involves l2,l3,lb"),
    ("mixed_fourth", (1, 1, 1, 2, 2, 2, 1, 0, 2, 0), LC_SUM,
     "fourth mixed term sums over (lc, mc) exactly like the second CV term"),

    # -- shotnoise: term 1 adds 1 at (la,ma,lb,mb)=(l1,m1,l2,m2); term 2 adds
    #    G(l1,l2,la,m1,m2,ma) at lb=mb=0. They collide whenever l2=m2=0. -------
    ("shotnoise", (0, 0, 0, 0, 0, 0, 0, 0), 1.0 + G_000,
     "l1=l2=0: unit term and Gaunt term collide in the same slot"),
    ("shotnoise", (1, 0, 2, 0, 1, 0, 2, 0), 1.0 + G_000,
     "l1=2, l2=0: collision again -> 1 + G(2,0,2,0,0,0)"),
    ("shotnoise", (1, 1, 2, 4, 1, 1, 2, 4), 1.0,
     "l2=2 so lb=2 != 0: unit term stands alone"),
]


@pytest.mark.parametrize(
    "array_name, index, expected, description",
    GAUNT_VALUE_CASES,
    ids=[f"{c[0]}-{c[3]}" for c in GAUNT_VALUE_CASES],
)
def test_gaunt_coefficient_values(gaunt_coefficients, array_name, index, expected, description):
    value = gaunt_coefficients[array_name][index]
    assert np.isclose(value, expected, rtol=0, atol=1e-13), \
        f"{array_name}{index} ({description}): got {value!r}, expected {expected!r}"

# Slots that are in range but must be exactly zero, either because the ell
# triangle condition fails or because of real-Gaunt m-selection.
GAUNT_ZERO_CASES = [
    ("first_cosmic_variance", (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0),
     "la=2 violates the triangle condition for l1=l4=0"),
    ("first_cosmic_variance", (0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0),
     "l2=2 must couple through lb, not la"),
    ("shotnoise", (0, 0, 0, 0, 1, 0, 2, 0),
     "la=2 violates the triangle condition for l1=l2=0"),
    ("shotnoise", (1, 1, 2, 4, 1, 0, 2, 0),
     "G(2,2,2,0,2,ma) vanishes unless |ma|=2"),
]


@pytest.mark.parametrize(
    "array_name, index, description",
    GAUNT_ZERO_CASES,
    ids=[f"{c[0]}-{c[2]}" for c in GAUNT_ZERO_CASES],
)
def test_gaunt_coefficient_structural_zeros(gaunt_coefficients, array_name, index, description):
    value = gaunt_coefficients[array_name][index]
    assert value == 0.0, f"{array_name}{index} ({description}): expected 0, got {value!r}"


@pytest.mark.parametrize("function, term", [
    ("first_cosmic_variance", None),
    ("second_cosmic_variance", None),
    ("mixed_term", "first"),
    ("mixed_term", "second"),
    ("mixed_term", "third"),
    ("mixed_term", "fourth"),
    ("shotnoise", None),
])
@pytest.mark.mpi(min_size=2)
def test_gaunt_coefficient_methods_are_mpi_safe(function, term):

    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    
    if rank == 0:
        for f in glob.glob(os.path.join(get_cache_dir(), "*coefficients*.npz")):
            os.remove(f)
    comm.Barrier()  # Ensure all processes wait for the file to be removed before proceeding

    if function == "first_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_first_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "second_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_second_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "mixed_term":
          coefficients = geometry.SurveyGeometry.get_mixed_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm, term=term)
    elif function == "shotnoise":
          coefficients = geometry.SurveyGeometry.get_shotnoise_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    
    comm.Barrier()
    
    # try accessing some properties of coefficients to ensure they were loaded correctly
    assert coefficients is not None
    idx_nonzero, values_nonzero = coefficients.T.get_nonzero_rows_dense()
    assert type(idx_nonzero) == np.ndarray
    assert type(values_nonzero) == np.ndarray
    total_iterations = len(idx_nonzero)
    assert type(total_iterations) == int
    assert total_iterations > 0

    # Now do the exact same tests, but without clearing the cache, so that we test the loading path instead of the computation path. This ensures both paths are MPI-safe.
    if function == "first_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_first_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "second_cosmic_variance":
          coefficients = geometry.SurveyGeometry.get_second_cosmic_variance_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    elif function == "mixed_term":
          coefficients = geometry.SurveyGeometry.get_mixed_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm, term=term)
    elif function == "shotnoise":
          coefficients = geometry.SurveyGeometry.get_shotnoise_gaunt_coefficients(
              mask_ellmax=2, pk_ellmax=2, cache_dir=get_cache_dir(), rank=rank, comm=comm)
    
    comm.Barrier()
    # try accessing some properties of coefficients to ensure they were loaded correctly
    assert coefficients is not None
    idx_nonzero, values_nonzero = coefficients.T.get_nonzero_rows_dense()
    assert type(idx_nonzero) == np.ndarray
    assert type(values_nonzero) == np.ndarray
    total_iterations = len(idx_nonzero)
    assert type(total_iterations) == int
    assert total_iterations > 0

    if rank == 0:
        for f in glob.glob(os.path.join(get_cache_dir(), "*coefficients*.npz")):
            os.remove(f)
    # mpirun -n 2 python -m pytest -v --capture=tee-sys --tb=short --with-mpi thecov/tests -m mpi


WINDOW_ELLMAX = 2   # -> 6 (l,m) pairs, so 6^4 = 1296 tuples for the 4-harmonic terms
WINDOW_NMESH = 6
WINDOW_KBINS = 5

# key -> (how many Ylm factors the term is a product of, which of those factors
# are evaluated along k2_hat). Mirrors the dispatch in compute_window_matrix; if
# that dispatch changes, these must change with it.
WINDOW_TERMS = {
    "first_cosmic_variance":  (4, (2, 3)),   # Y_k1(l1) Y_k1(l2) Y_k2(l3) Y_k2(l4)
    "second_cosmic_variance": (4, (1, 2)),   # Y_k1(l1) Y_k2(l2) Y_k2(l3) Y_k1(l4)
    "mixed_term":             (3, (1, 2)),   # Y_k1(l1) Y_k2(l2) Y_k2(l3)
    "shotnoise":              (2, (1,)),     # Y_k1(l1) Y_k2(l2)
}


def make_window_product(n_harmonics, populated_fraction=1.0, complex_data=False,
                        drop_fraction=0.0, seed=0):
    """Build a stand-in for the ``W @ G`` product: one dense mesh per (l, m) tuple.

    Only rows reachable from :func:`utils.ellmiter` are ever populated, and only
    ``populated_fraction`` of those -- the rest stand in for tuples whose Gaunt
    coefficients vanish, which the real product leaves structurally empty.

    ``drop_fraction`` punches exact zeros into the meshes. That is what knocks
    k1RowBookkeeping off its full-mesh fast path and onto the bincount fallback,
    since a row no longer covers every voxel.

    Returns
    -------
    window_product : base.SparseNDArray
    lm_tuples : list of tuple
        Every (l1..ln, m1..mn) tuple the term sums over, in ellmiter order.
    """
    rng = np.random.default_rng(seed)
    shape_out = n_harmonics * [WINDOW_ELLMAX // 2 + 1] + n_harmonics * [2 * WINDOW_ELLMAX + 1]
    shape_in = (WINDOW_NMESH, WINDOW_NMESH, WINDOW_NMESH)
    n_voxels = WINDOW_NMESH ** 3

    lm_tuples = list(utils.ellmiter(WINDOW_ELLMAX, n_harmonics))
    rows = np.array([np.ravel_multi_index(
        tuple(l // 2 for l in lm[:n_harmonics]) +
        tuple(m + l for m, l in zip(lm[n_harmonics:], lm[:n_harmonics])), shape_out)
        for lm in lm_tuples])
    populated = np.unique(rows[rng.random(len(rows)) < populated_fraction])

    indptr = np.zeros(int(np.prod(shape_out)) + 1, dtype=np.int64)
    indptr[populated + 1] = n_voxels
    indptr = np.cumsum(indptr)
    data = rng.standard_normal(len(populated) * n_voxels)
    if complex_data:
        # The real product can be complex; compute_window_matrix takes .real of it.
        data = data + 1j * rng.standard_normal(len(populated) * n_voxels)
    indices = np.tile(np.arange(n_voxels, dtype=np.int32), len(populated))

    matrix = scipy.sparse.csr_matrix((data, indices, indptr),
                                     shape=(int(np.prod(shape_out)), n_voxels))
    if drop_fraction:
        matrix.data[rng.random(matrix.nnz) < drop_fraction] = 0.0
        matrix.eliminate_zeros()

    window_product = base.SparseNDArray(shape_out, shape_in)
    # Assign directly rather than going through from_arrays, which populates the
    # matrix on the root rank only. These tests are not MPI tests and must hold on
    # every rank they happen to run on.
    window_product._matrix = matrix
    return window_product, lm_tuples


def make_Ylm_tables(seed=0):
    """Ylm(k1_hat) and Ylm(k2_hat) exactly as compute_window_matrix builds them.

    Ylm_k1 entries are scalars (one k1 mode); Ylm_k2 entries are meshes, one
    value per voxel -- except Y_00, which is constant and so comes back scalar.
    """
    rng = np.random.default_rng(seed)
    table = thecov_math.build_Ylm_table(WINDOW_ELLMAX)
    k1_hat = rng.standard_normal(3)
    k1_hat /= np.linalg.norm(k1_hat)
    k2_hat = rng.standard_normal((3, WINDOW_NMESH, WINDOW_NMESH, WINDOW_NMESH))
    k2_hat /= np.linalg.norm(k2_hat, axis=0)
    return (thecov_math.evaluate_Ylms(table, WINDOW_ELLMAX, *k1_hat),
            thecov_math.evaluate_Ylms(table, WINDOW_ELLMAX, *k2_hat))


def reduce_window_rows_reference(window_product, lm_tuples, k2_harmonics,
                                 Ylm_k1, Ylm_k2, k2_bin_index, kbins):
    """Transcription of the loop compute_window_matrix ran before the optimization.

    Densifies one mesh per (l, m) tuple, scales it by the Ylm factors and
    bincounts it into k2 bins. Deliberately naive -- this is the oracle.
    """
    n_harmonics = len(lm_tuples[0]) // 2
    n_ells = WINDOW_ELLMAX // 2 + 1
    result = np.zeros(n_harmonics * [n_ells] + [kbins])
    flat_bins = np.asarray(k2_bin_index).ravel()
    in_range = (flat_bins >= 0) & (flat_bins < kbins)

    for lm in lm_tuples:
        ells, ems = lm[:n_harmonics], lm[n_harmonics:]
        ell_idx = tuple(l // 2 for l in ells)
        em_idx = tuple(m + l for m, l in zip(ems, ells))
        # Indexed the same way the old loop did, with ND indices into shape_out.
        mesh = window_product[ell_idx + em_idx].real.toarray().reshape(window_product.shape_in)
        for i in range(n_harmonics):
            table = Ylm_k2 if i in k2_harmonics else Ylm_k1
            mesh = mesh * table[ell_idx[i]][em_idx[i]]
        if in_range.any():
            result[ell_idx] += np.bincount(flat_bins[in_range],
                                           weights=mesh.ravel()[in_range],
                                           minlength=kbins)[:kbins]
    return result


def assert_bookkeeping_matches_reference(key, populated_fraction=1.0, complex_data=False,
                                         drop_fraction=0.0, seed=0, expect_full_meshes=True):
    """Run k1RowBookkeeping and the naive oracle on the same inputs and compare."""
    n_harmonics, k2_harmonics = WINDOW_TERMS[key]
    window_product, lm_tuples = make_window_product(
        n_harmonics, populated_fraction=populated_fraction, complex_data=complex_data,
        drop_fraction=drop_fraction, seed=seed)
    Ylm_k1, Ylm_k2 = make_Ylm_tables(seed=seed)

    rng = np.random.default_rng(seed + 1000)
    # Spans past both ends of the k range so the out-of-range masking is exercised.
    k2_bin_index = rng.integers(-2, WINDOW_KBINS + 2,
                                size=(WINDOW_NMESH, WINDOW_NMESH, WINDOW_NMESH))

    bookkeeping = utils.k1RowBookkeeping(window_product, WINDOW_ELLMAX,
                                         n_harmonics, k2_harmonics)
    assert bookkeeping.rows_are_full_meshes == expect_full_meshes, \
        f"{key}: expected full-mesh fast path = {expect_full_meshes}"

    got = bookkeeping.reduce(window_product._matrix, Ylm_k1, Ylm_k2,
                             k2_bin_index, WINDOW_KBINS)
    expected = reduce_window_rows_reference(window_product, lm_tuples, k2_harmonics,
                                            Ylm_k1, Ylm_k2, k2_bin_index, WINDOW_KBINS)

    assert got.shape == expected.shape
    scale = np.abs(expected).max()
    assert scale > 0, f"{key}: oracle produced an all-zero result, test is vacuous"
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9 * scale), \
        f"{key}: max relative deviation {np.abs(got - expected).max() / scale:.2e}"
    return bookkeeping


@pytest.mark.parametrize("key", list(WINDOW_TERMS))
def test_k1_bookkeeping_basic(key):

    # Test default case
    assert_bookkeeping_matches_reference(key)

    # Test with a partially populated window
    bookkeeping = assert_bookkeeping_matches_reference(key, populated_fraction=0.4, seed=7)
    assert len(bookkeeping.mesh_rows) < bookkeeping.n_lm_tuples, \
        "test did not actually produce any empty rows"

    # Now test that the bookkeeping can handle complex data
    assert_bookkeeping_matches_reference(key, complex_data=True, seed=3)


@pytest.mark.parametrize("key", list(WINDOW_TERMS))
def test_k1_bookkeeping_fallback_when_window_has_exact_zeros(key):
    """Exact zeros in the window force the bincount fallback, which must still agree.

    A row that no longer covers every voxel can't be read straight out of the CSR
    data array, so k1RowBookkeeping honours the column indices instead.
    """
    assert_bookkeeping_matches_reference(key, drop_fraction=0.3, seed=5,
                                        expect_full_meshes=False)


@pytest.mark.parametrize("key", list(WINDOW_TERMS))
def test_k1_bookkeeping_returns_zeros_when_no_k2_bin_in_range(key):
    """A k1 mode whose whole k2 mesh falls outside the k range contributes nothing."""
    n_harmonics, k2_harmonics = WINDOW_TERMS[key]
    window_product, _ = make_window_product(n_harmonics)
    Ylm_k1, Ylm_k2 = make_Ylm_tables()
    bookkeeping = utils.k1RowBookkeeping(window_product, WINDOW_ELLMAX,
                                         n_harmonics, k2_harmonics)

    n_ells = WINDOW_ELLMAX // 2 + 1
    for out_of_range in (-5, WINDOW_KBINS + 5):
        bins = np.full((WINDOW_NMESH, WINDOW_NMESH, WINDOW_NMESH), out_of_range)
        got = bookkeeping.reduce(window_product._matrix, Ylm_k1, Ylm_k2, bins, WINDOW_KBINS)
        assert got.shape == n_harmonics * (n_ells,) + (WINDOW_KBINS,)
        assert np.all(got == 0.0), f"{key}: expected no contribution for bins={out_of_range}"


def test_evaluate_Ylms_returns_scalar_for_the_monopole():
    """Y_00 is constant over the sphere, so evaluate_Ylms returns a scalar, not a mesh.

    k1RowBookkeeping._pack_Ylm_table has to broadcast that scalar across the voxel
    axis rather than reshape it. This pins the assumption: if a sympy/pypower
    change ever makes evaluate_Ylms return a full mesh for Y_00 (or makes some
    other (l, m) collapse to a scalar), this test says so directly instead of the
    failure surfacing as a reshape error deep in the window matrix loop.
    """
    _, Ylm_k2 = make_Ylm_tables()
    n_voxels = WINDOW_NMESH ** 3

    assert np.size(Ylm_k2[0][0]) == 1, "expected Y_00 to be constant over the mesh"
    assert np.isclose(np.asarray(Ylm_k2[0][0]).item(), 1.0 / np.sqrt(4 * np.pi))

    for ell_i, ell in enumerate(range(2, WINDOW_ELLMAX + 1, 2), start=1):
        for em_i in range(2 * ell + 1):
            assert np.size(Ylm_k2[ell_i][em_i]) == n_voxels, \
                f"expected Y_{ell},{em_i - ell} to vary over the mesh"