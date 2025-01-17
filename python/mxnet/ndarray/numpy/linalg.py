# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Namespace for operators used in Gluon dispatched by F=ndarray."""

from __future__ import absolute_import
from . import _op as _mx_nd_np
from . import _internal as _npi

__all__ = ['norm', 'svd', 'cholesky', 'inv', 'det', 'slogdet']


def norm(x, ord=None, axis=None, keepdims=False):
    r"""Matrix or vector norm.

    This function can only support Frobenius norm for now.
    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Parameters
    ----------
    x : ndarray
        Input array.
    ord : {'fro'}, optional
        Order of the norm.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None, the norm of the whole ndarray is
        returned.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    """
    if ord is not None and ord != 'fro':
        raise ValueError('only support Frobenius norm for now, received ord={}'.format(str(ord)))
    if isinstance(axis, tuple) and len(axis) > 2:
        raise ValueError('Improper number of dimensions to norm')
    if ord == 'fro' and x.ndim > 2 and axis is None:
        raise ValueError('Improper number of dimensions to norm')
    return _mx_nd_np.sqrt(_mx_nd_np.sum(x * x, axis=axis, keepdims=keepdims))


def svd(a):
    r"""
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) ndarray
        A real array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    ut: (..., M, M) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    s : (..., M) ndarray
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    v : (..., M, N) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.

    Notes
    -----

    The decomposition is performed using LAPACK routine ``_gesvd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U^T S V`, where :math:`A = a`, :math:`U^T = ut`,
    :math:`S= \mathtt{np.diag}(s)` and :math:`V = v`. The 1D array `s`
    contains the singular values of `a` and `ut` and `v` are orthonormal. The rows
    of `v` are the eigenvectors of :math:`A^T A` and the columns of `ut` are
    the eigenvectors of :math:`A A^T`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    The sign of rows of `u` and `v` are determined as described in
    `Auto-Differentiating Linear Algebra <https://arxiv.org/pdf/1710.08717.pdf>`_.

    If `a` has more than two dimensions, then broadcasting rules apply.
    This means that SVD is working in "stacked" mode: it iterates over
    all indices of the first ``a.ndim - 2`` dimensions and for each
    combination SVD is applied to the last two indices. The matrix `a`
    can be reconstructed from the decomposition with either
    ``(ut * s[..., None, :]) @ v`` or
    ``ut @ (s[..., None] * v)``. (The ``@`` operator denotes batch matrix multiplication)

    This function differs from the original `numpy.linalg.svd
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html>`_ in
    the following way(s):
     - The sign of rows of `u` and `v` may differ.
     - Does not support complex input.

    Examples
    --------
    >>> a = np.arange(54).reshape(6, 9)
    >>> ut, s, v = np.linalg.svd(a)
    >>> ut.shape, s.shape, v.shape
    ((6, 6), (6,), (6, 9))
    >>> s = s.reshape(6, 1)
    >>> ret = np.dot(ut, s * v)
    >>> (ret - a > 1e-3).sum()
    array(0.)
    >>> (ret - a < -1e-3).sum()
    array(0.)
    """
    return tuple(_npi.svd(a))


def cholesky(a):
    r"""
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.T`, of the square matrix `a`,
    where `L` is lower-triangular and .T is the transpose operator. `a` must be
    symmetric and positive-definite. Only `L` is actually returned. Complex-valued
    input is currently not supported.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Symmetric, positive-definite input matrix.

    Returns
    -------
    L : (..., M, M) ndarray
        Lower-triangular Cholesky factor of `a`.

    Raises
    ------
    MXNetError
        If the decomposition fails, for example, if `a` is not positive-definite.

    Notes
    -----
    Broadcasting rules apply.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \mathbf{x} = \mathbf{b}

    (when `A` is both symmetric and positive-definite).

    First, we solve for :math:`\mathbf{y}` in

    .. math:: L \mathbf{y} = \mathbf{b},

    and then for :math:`\mathbf{x}` in

    .. math:: L.T \mathbf{x} = \mathbf{y}.

    Examples
    --------
    >>> A = np.array([[16, 4], [4, 10]])
    >>> A
    array([[16.,  4.],
           [ 4., 10.]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[4., 0.],
           [1., 3.]])
    >>> np.dot(L, L.T)
    array([[16.,  4.],
           [ 4., 10.]])
    """
    return _npi.cholesky(a)


def inv(a):
    r"""
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    MXNetError
        If `a` is not square or inversion fails.

    Examples
    --------
    >>> from mxnet import np
    >>> a = np.array([[1., 2.], [3., 4.]])
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])

    Inverses of several matrices can be computed at once:

    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> np.linalg.inv(a)
    array([[[-2.        ,  1.        ],
            [ 1.5       , -0.5       ]],

           [[-1.2500001 ,  0.75000006],
            [ 0.75000006, -0.25000003]]])
    """
    return _npi.inv(a)


def det(a):
    r"""
    Compute the determinant of an array.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Input array to compute determinants for.

    Returns
    -------
    det : (...) ndarray
        Determinant of `a`.

    See Also
    --------
    slogdet : Another way to represent the determinant, more suitable
    for large matrices where underflow/overflow may occur.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.det(a)
    -2.0

    Computing determinants for a stack of matrices:
    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)

    >>> np.linalg.det(a)
    array([-2., -3., -8.])
    """
    return _npi.det(a)


def slogdet(a):
    r"""
    Compute the sign and (natural) logarithm of the determinant of an array.
    If an array has a very small or very large determinant, then a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Input array, has to be a square 2-D array.

    Returns
    -------
    sign : (...) ndarray
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1.
    logdet : (...) array_like
        The natural log of the absolute value of the determinant.
    If the determinant is zero, then `sign` will be 0 and `logdet` will be
    -Inf. In all cases, the determinant is equal to ``sign * np.exp(logdet)``.

    See Also
    --------
    det

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    Examples
    --------
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:
    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logdet) = np.linalg.slogdet(a)
    >>> (sign, logdet)
    (-1., 0.69314718055994529)

    >>> sign * np.exp(logdet)
    -2.0

    Computing log-determinants for a stack of matrices:
    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)

    >>> sign, logdet = np.linalg.slogdet(a)
    >>> (sign, logdet)
    (array([-1., -1., -1.]), array([ 0.69314718,  1.09861229,  2.07944154]))

    >>> sign * np.exp(logdet)
    array([-2., -3., -8.])

    This routine succeeds where ordinary `det` does not:
    >>> np.linalg.det(np.eye(500) * 0.1)
    0.0
    >>> np.linalg.slogdet(np.eye(500) * 0.1)
    (1., -1151.2925464970228)
    """
    return _npi.slogdet(a)
