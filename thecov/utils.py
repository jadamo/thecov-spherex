"""This module contains utility functions for thecov.
"""
import os, functools, psutil
import numpy as np
import scipy
import itertools as itt
from scipy.interpolate import InterpolatedUnivariateSpline

def mkdir(dirname):
    """Try to create ``dirname`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname)  # MPI...
    except OSError:
        return
    
# python's enumerate but with a custom step = 2
def enum2(xs, start=0, step=2):
    """Enumerate a sequence with a custom step.

    Parameters
    ----------
    xs : sequence
        Sequence to enumerate.
    start : int, optional
        Starting index. Default is 0.
    step : int, optional
        Step of the enumeration. Default is 2.

    Returns
    -------
    generator
        Generator of tuples (index, element).
    """
    for x in xs:
        yield (start, x)
        start += step

def limit(iterable, count):
    """
    Limit number of iterated elements from an iterable.
    count -- is the maximum number of elements to iterate through
    """
    while count > 0:
        yield next(iterable)
        count -= 1

def cache_method(func):
    '''Decorator to cache the result of a method.

    Parameters
    ----------
    func : callable
        Method to cache.

    Returns
    -------
    callable
        Cached method.
    '''

    @functools.wraps(func)
    def cached_func(self, *args, **kwargs):
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = {}
        
        if len(args) + len(kwargs) == 1:
            key = args[0] if args else next(iter(kwargs.values()))
        else:
            key = hash((args, frozenset(kwargs.items())))

        cache = self._cache[func.__name__]
        
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
    
        return cache[key]

    return cached_func

def ellmiter(lmax, n):
    """Creates generator over all combinations of ell and m up to lmax for n tracers."""
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        for ms in itt.product(*[range(-l, l+1, 1) for l in ls]):
            yield ls + ms

def mpi_ellmiter(lmax, n, comm):
    """Like ellmiter, but each MPI rank yields only its assigned subset of (l, m) tuples.

    Distributes the outer ell-tuple loop round-robin across ranks so that
    heavy (high-ell) blocks are spread evenly.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i, ls in enumerate(itt.product(range(0, lmax + 1, 2), repeat=n)):
        if i % size == rank:
            for ms in itt.product(*[range(-l, l+1, 1) for l in ls]):
                yield ls + ms
                
def elliter(lmax, n):
    """Creates generator over all combinations of ell up to lmax for n tracers."""
    for ls in itt.product(range(0, lmax + 1, 2), repeat=n):
        yield ls


def miter(*ls):
    for ms in itt.product(*[range(-l, l+1, 1) for l in ls]):
        yield ms

def get_tqdm():
    """Get the tqdm module, compatible with Jupyter notebooks and terminals."""
    try: 
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            from tqdm.notebook import tqdm as tqdm
        else:
            # Terminal or other environment
            from tqdm import tqdm as tqdm
    except NameError:
        # Not in a Jupyter environment
        from tqdm import tqdm as tqdm
    return tqdm

def get_minimum_mesh_size(dk, kmax, boxsize):
    """Get the minimum mesh size for a given dk, kmax, and boxsize."""
    target_boxsize = 2*np.pi/dk
    min_nmesh = (target_boxsize * kmax / np.pi) / (target_boxsize/boxsize)
    return int(np.ceil(min_nmesh))

def get_available_memory():
    """Get the available system memory in Gigabytes."""
    return psutil.virtual_memory().available / (1024 ** 3)


def gather_field_to_root(field, root=0):
    """Gather a distributed 3D slab `field` onto `root`, preserving spatial layout.

    Parameters
    ----------
    field : pmesh Field
        A pmesh Field (e.g. `RealField`) with attributes `pm`, `start`, `shape`, and
        `value` representing the local slab (numpy array) on each rank.
    root : int
        MPI rank to gather to. Default is 0.

    Returns
    -------
    numpy.ndarray or None
        On `root`, returns the reconstructed full array with global shape
        `field.pm.Nmesh`. On non-root ranks, returns ``None``.
    """
    import numpy as _np

    pm = field.pm
    comm = pm.comm

    # local slab and its global start/shape
    local = _np.array(field.value, copy=False)
    start = tuple(int(s) for s in field.start)
    shape = tuple(int(s) for s in field.shape)

    # gather starts and shapes from all ranks to the root
    all_starts = comm.gather(start, root=root)
    all_shapes = comm.gather(shape, root=root)
    all_slabs = comm.gather(local, root=root)

    if comm.rank != root:
        return 0 # <- dummy number to avoid NoneType issues

    # allocate full array on root
    # use Nmesh for real-space fields, for complex fields use pm.Nmesh but their
    # represented storage may differ. We'll use pm.Nmesh for spatial layout.
    full_shape = tuple(int(n) for n in pm.Nmesh)
    full = _np.zeros(full_shape, dtype=local.dtype)

    # place each slab into the full array at the recorded start
    for st, sh, slab in zip(all_starts, all_shapes, all_slabs):
        slices = tuple(slice(s, s + n) for s, n in zip(st, sh))
        full[slices] = slab

    return full

def trim_fourier_mesh(mesh:np.ndarray, nmesh:int, new_nmesh:int):
    """Trim a fourier-space mesh to a smaller size, preserving positive and negative frequencies.

    Parameters
    ----------
    mesh : numpy.ndarray
        The fourier-space mesh to trim.
    nmesh : int
        The original size of the mesh.
    new_nmesh : int
        The desired size of the mesh.

    Returns
    -------
    numpy.ndarray
        The trimmed fourier-space mesh.
    """
    mesh = np.fft.fftshift(mesh)
    center = nmesh // 2
    half   = new_nmesh // 2
    mesh = mesh[center-half:center+half,
                center-half:center+half,
                center-half:center+half]
    return np.fft.ifftshift(mesh)

def build_radial_profile(positions, values, comm, n_bins=500):
    """Creats a radial profile interpolator (MPI-safe) from a set of 3D positions and values.
 
    Each rank contributes its local particles; bin sums and counts are
    reduced across all ranks before building the spline.
 
    Parameters
    ----------
    positions : array_like, shape (N_local, 3)
        Local Cartesian positions (observer at origin).
    values : array_like, shape (N_local,)
        Local quantity to profile.
    comm : MPI communicator
        MPI communicator.
    n_bins : int
        Number of radial bins.
 
    Returns
    -------
    InterpolatedUnivariateSpline
        Spline interpolator for the radial profile (identical on all ranks).
    """
    from mpi4py import MPI
 
    r = np.sqrt(np.sum(positions**2, axis=-1))
 
    # Global min/max for consistent bin edges across ranks
    local_bounds = np.array([r.min(), r.max()])
    global_bounds = np.empty(2)
    comm.Allreduce(np.array([r.min()]), global_bounds[:1], op=MPI.MIN)
    comm.Allreduce(np.array([r.max()]), global_bounds[1:], op=MPI.MAX)
 
    r_min, r_max = global_bounds
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1)
 
    # Local bin sums and counts
    local_sums = np.bincount(bin_idx, weights=values, minlength=n_bins).astype(np.float64)
    local_counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
 
    # Reduce across ranks
    global_sums = np.empty_like(local_sums)
    global_counts = np.empty_like(local_counts)
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
 
    valid = global_counts > 0
    bin_avg = global_sums[valid] / global_counts[valid]
 
    return InterpolatedUnivariateSpline(bin_centers[valid], bin_avg, k=3)
 
 
def interpolate_to_positions(profile, positions):
    """Evaluate a radial profile at given positions.
 
    Parameters
    ----------
    profile : InterpolatedUnivariateSpline
        Radial profile from build_radial_profile.
    positions : array_like, shape (M, 3)
        Cartesian positions where values are needed.
 
    Returns
    -------
    array_like, shape (M,)
        Interpolated values.
    """
    r = np.sqrt(np.sum(positions**2, axis=-1))
    return profile(r)

class k1RowBookkeeping:
    r"""Bins the window product over k2 for one sampled k1 mode at a time.

    Each term of the window matrix is a sum over the voxels of a k2 mesh of the
    window product times a product of ``n_harmonics`` spherical harmonics, some
    evaluated along `\hat{k}_1` and some along :math:`\hat{k}_2`:

    Every sampled k1 mode reads the same set of :math:`(\ell, m)` tuples out of
    ``window_product``, so the constructor precomputes everything that does not
    depend on the mode:

    * the flat CSR row holding each tuple's mesh,
    * which tuples map to structurally empty rows (they contribute nothing and are
      dropped rather than densified),
    * an ordering that visits tuples sharing the same :math:`\hat{k}_2` harmonic
      indices consecutively, so their :math:`Y_{\ell m}(\hat{k}_2)` mesh product is
      formed once per group instead of once per tuple.

    :meth:`reduce` then does the per-mode work.

    Parameters
    ----------
    window_product : base.SparseNDArray
        The :math:`W \otimes G` product being reduced. Only its shapes and CSR
        ``indptr`` are read here.
    pk_ellmax : int
        Maximum power spectrum multipole.
    n_harmonics : int
        How many spherical harmonics the term is a product of, i.e. how many
        :math:`(\ell, m)` pairs index ``window_product``'s outer shape. 4 for the
        cosmic variance terms, 3 for the mixed terms, 2 for shotnoise.
    k2_harmonics : sequence of int
        Positions in the :math:`(\ell_1 m_1, \ldots, \ell_n m_n)` tuple whose
        harmonic is evaluated along :math:`\hat{k}_2`. The remaining positions are
        evaluated along :math:`\hat{k}_1`. For example the first cosmic variance
        term has :math:`Y_{\ell_1 m_1}(\hat{k}_1) Y_{\ell_2 m_2}(\hat{k}_1)
        Y_{\ell_3 m_3}(\hat{k}_2) Y_{\ell_4 m_4}(\hat{k}_2)`, so ``(2, 3)``.
    """

    def __init__(self, window_product, pk_ellmax, n_harmonics, k2_harmonics):
        self.pk_ellmax = pk_ellmax
        self.n_harmonics = n_harmonics
        self.n_ells = pk_ellmax // 2 + 1          # number of even ell values
        self.n_ems = 2 * pk_ellmax + 1            # widest m range, for m = -l..l
        self.k2_harmonics = list(k2_harmonics)
        self.k1_harmonics = [i for i in range(n_harmonics) if i not in self.k2_harmonics]
        self.n_voxels = int(np.prod(window_product.shape_in))
        self.n_output_rows = self.n_ells ** n_harmonics

        # All (l1..ln, m1..mn) tuples the term sums over, split into array indices:
        # ell_indices = l/2 (only even ell), em_indices = m + l (shifts m to 0..2l).
        lm_tuples = np.array(list(ellmiter(pk_ellmax, n_harmonics)), dtype=int)
        self.n_lm_tuples = len(lm_tuples)
        ells, ems = lm_tuples[:, :n_harmonics], lm_tuples[:, n_harmonics:]
        ell_indices = ells // 2
        em_indices = ems + ells

        # The CSR row of window_product holding each tuple's mesh.
        shape_out = tuple(int(s) for s in window_product.shape_out)
        mesh_rows = np.ravel_multi_index(tuple(ell_indices.T) + tuple(em_indices.T), shape_out)

        indptr = window_product._matrix.indptr
        row_nnz = indptr[mesh_rows + 1] - indptr[mesh_rows]
        is_populated = row_nnz > 0      # empty rows come from vanishing Gaunt coefficients

        # Group tuples by their k2-side harmonic indices, so all tuples in a group
        # share one Ylm(k2) mesh product. The group key packs those (l, m) indices
        # into a single sortable integer.
        k2_group_key = np.zeros(len(lm_tuples), dtype=np.int64)
        for harmonic in self.k2_harmonics:
            k2_group_key = ((k2_group_key * self.n_ells + ell_indices[:, harmonic])
                            * self.n_ems + em_indices[:, harmonic])
        # Sort by group, pushing the empty rows to the end so they can be sliced off.
        row_order = np.argsort(np.where(is_populated, k2_group_key, k2_group_key.max() + 1),
                               kind='stable')
        row_order = row_order[is_populated[row_order]]

        self.mesh_rows = mesh_rows[row_order]
        self.ell_indices = ell_indices[row_order]
        self.em_indices = em_indices[row_order]
        self.row_nnz = row_nnz[row_order]
        # Where each tuple's contribution lands in the flattened (l1, ..., ln) output.
        self.output_rows = np.ravel_multi_index(tuple(self.ell_indices.T),
                                                n_harmonics * (self.n_ells,))
        # A row holding a full mesh in canonical CSR order covers voxels 0..n_voxels-1
        # in order, so its values can be read straight out of the CSR data array. If
        # the window has exact zeros in it, fall back to a per-row bincount that
        # respects the column indices.
        self.rows_are_full_meshes = bool(len(self.mesh_rows)
                                         and np.all(self.row_nnz == self.n_voxels)
                                         and window_product._matrix.has_sorted_indices)

        # Row index where each group starts, with a closing sentinel at the end.
        if len(row_order):
            group_edges = np.flatnonzero(np.r_[True, np.diff(k2_group_key[row_order]) != 0])
            self.group_starts = np.r_[group_edges, len(row_order)]
        else:
            self.group_starts = np.zeros(1, dtype=int)

    def _pack_Ylm_table(self, Ylm, value_shape=()):
        """Copy the ragged ``Ylm[l][m]`` list into one ``[n_ells, n_ems, *value_shape]`` array.

        Lets the (l, m) indices be used as array indices, so the harmonics for many
        (l, m) tuples can be looked up in a single vectorized gather. ``value_shape``
        is ``()`` for the scalar Ylm(k1) and ``(n_voxels,)`` for the Ylm(k2) meshes.
        """
        packed = np.zeros((self.n_ells, self.n_ems) + tuple(value_shape))
        for ell_i, ell in enumerate(range(0, self.pk_ellmax + 1, 2)):
            for em_i in range(2 * ell + 1):
                packed[ell_i, em_i] = np.reshape(Ylm[ell_i][em_i], value_shape)
        return packed

    def reduce(self, product_matrix, Ylm_k1, Ylm_k2, k2_bin_index, kbins):
        """Accumulate one k1 mode's contribution, binned in k2.

        Parameters
        ----------
        product_matrix : scipy.sparse.csr_matrix
            The ``window_product`` CSR matrix, one mesh per row.
        Ylm_k1, Ylm_k2 : list of list
            Harmonics from :func:`thecov.math.evaluate_Ylms`, indexed ``[l//2][m+l]``.
            ``Ylm_k1`` entries are scalars, ``Ylm_k2`` entries are meshes.
        k2_bin_index : numpy.ndarray
            k-bin each mesh voxel falls into. Values outside ``[0, kbins)`` are
            outside the k range of the covariance and are dropped.
        kbins : int
            Number of k bins.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_ells,) * n_harmonics + (kbins,)``, to be added into
            the window matrix at the current k1 bin.
        """
        output_shape = self.n_harmonics * (self.n_ells,) + (kbins,)
        contribution = np.zeros((self.n_output_rows, kbins))
        if not len(self.mesh_rows):
            return contribution.reshape(output_shape)

        k2_bin_of_voxel = np.asarray(k2_bin_index).ravel()
        in_range = (k2_bin_of_voxel >= 0) & (k2_bin_of_voxel < kbins)
        if not in_range.any():
            return contribution.reshape(output_shape)

        Ylm_k1_table = self._pack_Ylm_table(Ylm_k1)
        Ylm_k2_table = self._pack_Ylm_table(Ylm_k2, value_shape=(self.n_voxels,))

        # The Ylm(k1) harmonics are scalars for this mode, so their product is one
        # number per (l, m) tuple -- gather them for all tuples at once.
        k1_prefactor = np.ones(len(self.mesh_rows))
        for harmonic in self.k1_harmonics:
            k1_prefactor *= Ylm_k1_table[self.ell_indices[:, harmonic],
                                         self.em_indices[:, harmonic]]

        data, indptr, indices = product_matrix.data, product_matrix.indptr, product_matrix.indices

        if self.rows_are_full_meshes:
            # Recast the k2 sum as a matrix-vector product. k2_binning_matrix has
            # shape [kbins, n_voxels] with one entry per in-range voxel, holding that
            # voxel's Ylm(k2) weight in the row of the bin it falls into. Then
            # `k2_binning_matrix @ mesh` applies the Ylm(k2) weighting, drops the
            # out-of-range voxels and sums into k2 bins in a single sparse kernel.
            #
            # Which voxel sits in which bin depends only on k2_bin_index, so the
            # sparsity pattern is built once here and only the weights (.data) are
            # refreshed per group. CSR wants its entries ordered by row, i.e. by bin.
            voxels_in_range = np.flatnonzero(in_range)
            voxels_by_bin = voxels_in_range[
                np.argsort(k2_bin_of_voxel[voxels_in_range], kind='stable')].astype(np.int32)
            voxels_per_bin = np.bincount(k2_bin_of_voxel[voxels_in_range], minlength=kbins)
            k2_binning_matrix = scipy.sparse.csr_matrix(
                (np.empty(len(voxels_by_bin)), voxels_by_bin,
                 np.r_[0, np.cumsum(voxels_per_bin)].astype(np.int32)),
                shape=(kbins, self.n_voxels))
        else:
            # Send out-of-range voxels to an overflow bin that gets sliced off. Cheaper
            # than boolean-masking the mesh on every row.
            overflow_bin = np.where(in_range, k2_bin_of_voxel, kbins).astype(np.intp)

        for group in range(len(self.group_starts) - 1):
            group_start, group_end = self.group_starts[group], self.group_starts[group + 1]

            # Product of the Ylm(k2) harmonics -- a mesh, shared by the whole group.
            k2_weight_mesh = np.ones(self.n_voxels)
            for harmonic in self.k2_harmonics:
                k2_weight_mesh = k2_weight_mesh * Ylm_k2_table[
                    self.ell_indices[group_start, harmonic],
                    self.em_indices[group_start, harmonic]]

            if self.rows_are_full_meshes:
                k2_binning_matrix.data = k2_weight_mesh[voxels_by_bin]
                for i in range(group_start, group_end):
                    mesh_start = indptr[self.mesh_rows[i]]
                    mesh = data[mesh_start:mesh_start + self.n_voxels].real
                    contribution[self.output_rows[i]] += k1_prefactor[i] * (k2_binning_matrix @ mesh)
            else:
                for i in range(group_start, group_end):
                    mesh_start, mesh_end = indptr[self.mesh_rows[i]], indptr[self.mesh_rows[i] + 1]
                    voxels = indices[mesh_start:mesh_end]
                    contribution[self.output_rows[i]] += k1_prefactor[i] * np.bincount(
                        overflow_bin[voxels],
                        weights=data[mesh_start:mesh_end].real * k2_weight_mesh[voxels],
                        minlength=kbins + 1)[:kbins]

        return contribution.reshape(output_shape)
