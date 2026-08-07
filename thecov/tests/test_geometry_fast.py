import numpy as np
import pytest
import logging
from thecov import geometry
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