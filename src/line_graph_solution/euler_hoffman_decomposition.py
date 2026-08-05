# The angles are unstable!!! There are two problems I recognised:
# - (Effectively) block diagonal matrices:
#   In the case of block diagonal matrices we need to ensure that each block has
#   determinant +1. If not, the decomposition can fail (mirrored signs), or alternatively,
#   we get non identity rotations that are effectively mirror operations that connect two
#   blocks, which we cannot allow for blocks induced by the connected components of the
#   line graph. We can easily deal with this "big" induced blocks by the frustration graph
#   by simply getting the angles for each block (and ensuring that each block has
#   determinant +1); this is already done in the code. However, for smaller blocks, or
#   more general blocks that would appear after a permutation conjugation, that can just
#   appear by chance and are not captured as components in the line graph, we would have
#   to detect these effective blocks. This could be done by defining an adjacency matrix
#   which has zeros where our original matrix has zeros (up to some threshold) and ones
#   elsewhere, and then getting the connected components of this adjacency matrix; this is
#   currently not done in the code.
#   About the permutations; this is actually a bit tricky and requires some additional
#   work in the theoretical diagonalisation of the Majorana Hamiltonian: ... I'm actually
#   not sure whether it is possible at all! The problem is that the new K' does not
#   (necessarily) commute through a the permutation matrix that would sandwich the h
#   matrix in the Hamiltonian....
# - Close to zero values in the matrix:
#   They just cause numerical instabilities. I tried a few things to mitigate this,
#   including setting theta angles to zero as described in the Hoffman paper (even if the
#   condition is only approximately met), but this
#   only made things worse (maybe because the matrix was "less" orthogonal afterwards;
#   maybe there is a small mistake or typo in the paper). The problem is induced by the
#   small values coming from the Schur decomposition, i.e., the Schur matrices have small
#   values where there (probably) should be zeros. I couldn't find a convenient library
#   with a more precise Schur decomposition (mpmath has only the complex Schur
#   decomposition). So either we implement a high-precision Schur decomposition ourselves,
#   or we find a Lapack implementation that supports higher than double float precision
#   and do some disgusting C to Python linking and binding.
#
# EDIT!!!: I believe the problems are actually connected, more specifically only the
# "effective blocks cause issues" and the close-to-zero issues is actually the same case,
# but not as bad because the would-be zero blocks are not exactly zero
# -> temporary solution: if a decomposition fails, we add small values (1e-10) to all zero
# small values in the matrix
# -> permanent solution: somehow fix the block problem (permutations seem not to work...,
# see above); and then maybe for the "close-to-zero" problem, just set those small values
# to zero...
#
# Note that the tests at the bottom with random orthogonal matrices work fine, as it is
# very unlikely that we encounter any (of the above) edge cases there.


from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union
from helper import numpy as np_helper


# keep the commented warning prints here, so that if something goes wrong in the future,
# they might give us a hint why
def arctan(opposite, nom, denom) -> float:
    # if np.isclose(denom, 0.0):  # makes things worse compared to ==0.0
    if denom == 0.0:
        # fmt: off
#         print(
# """warning; hoffman_angles: zero division case that is not directly handled in the Hoffman
# paper; double check correctness of the decomposition!"""
#         )
        # fmt: on
        # if np.isclose(nom, 0.0):
        if nom == 0.0:
            adjacent = 0.0
            # I think the next two cases should never happen...  edit: it can due to some
            # numerical inaccuracies -> check that it is close to zero
        elif nom < 0:
            assert np.isclose(nom, 0.0)
            # print("warning; hoffman_angles: -inf case")
            # adjacent = -np.inf
            # the above would gives theta = \pm pi which we would set to 0 anyways later
            # when we check for -pi/2 <= theta <= pi/2 (by adding/subtracting pi)
            adjacent = 0.0
        elif nom > 0:
            assert np.isclose(nom, 0.0)
            # print("warning; hoffman_angles: +inf case")
            # adjacent = np.inf
            # cf. nom < 0 case above
            adjacent = 0.0
        else:
            raise RuntimeError("unreachable")
    else:
        adjacent = nom / denom
    # if np.isclose(denom, 0.0):
    #     print(
    #         f"warning; hoffman_angles: small value division case: denom={denom}, nom={nom}"
    #     )
    if adjacent == 0.0 and np.isclose(opposite, 0.0):
        theta = 0.0
    else:
        theta = np.arctan2(opposite, adjacent)
    return theta


def get_angles(tm: NDArray[np.float64]):
    """
    note that the returned angles are kinda reversed, i.e., we return [[], [theta_1^2],
    [theta_1^3, theta_2^3], ..., [theta_1^n, ..., theta_{n-1}^n]]

    input the matrix transposed and set transpose(d)=True in get_matrices and
    reconstruct_matrix to have the decomposition work as multiplied from the left, i.e.,
    if tm = K^T, then K = R^2_1 @ R^3_1 @ R^3_2 @ ... @ R^n_1 @ ... @ R^n_{n-1}
    """
    n = tm.shape[0]
    assert tm.shape == (n, n)
    angles = []
    t_nu = tm.copy()

    for nu in range(n, 1, -1):
        angles_nu = []
        nu_idx = nu - 1

        k = nu - 1
        k_idx = k - 1
        opposite = t_nu[k_idx, nu_idx]
        adjacent = t_nu[k_idx + 1, nu_idx]
        theta = np.arctan2(opposite, adjacent)
        assert theta >= -np.pi and theta <= np.pi
        if np.isnan(theta):
            raise ValueError("NAN CASE ENCOUNTERED IN HOFFMAN_ANGLES")
        if not np.sign(np.sin(theta)) == np.sign(opposite):
            assert np.isclose(np.sin(theta), 0.0)
            assert np.isclose(opposite, 0.0)
        if not np.sign(np.cos(theta)) == np.sign(adjacent):
            if adjacent == 0 and opposite == 0:
                # special case ...
                assert np.cos(theta) == 1.0
            else:
                assert np.isclose(np.cos(theta), 0.0)
                assert np.isclose(adjacent, 0.0)
        angles_nu.append(theta)
        for k in range(nu - 2, 0, -1):
            k_idx = k - 1
            opposite = t_nu[k_idx, nu_idx]
            nom = t_nu[k_idx + 1, nu_idx]
            denom = np.sin(angles_nu[-1])
            theta = arctan(opposite, nom, denom)
            # the following +/- pi/2 changes should not happen  (it should only happen if
            # we get \pm inf in the arctan function for the "adjacent" value, which should
            # only happen because of numerical inaccuracies (we addressed that inaccuracy
            # there so this here should not happen))
            if not theta >= -np.pi / 2:
                theta += np.pi
                print("CHANGE!!! should not happen")
            if not theta <= np.pi / 2:
                print(theta)
                theta -= np.pi
                print("CHANGE!!! should not happen")
            assert theta >= -np.pi / 2 and theta <= np.pi / 2
            if not np.sign(np.sin(theta)) == np.sign(opposite):
                assert np.isclose(np.sin(theta), 0.0)
                assert np.isclose(opposite, 0.0)
            if np.isnan(theta):
                raise ValueError("NAN CASE ENCOUNTERED IN HOFFMAN_ANGLES")
            angles_nu.append(theta)
        angles_nu.reverse()
        angles.append(angles_nu)

        if nu > 2:
            f_nu = np.zeros((nu, nu), dtype=np.float64)
            new_t_nu = np.zeros((nu, nu), dtype=np.float64)
            new_t_nu[nu_idx, nu_idx] = 1.0
            for col in range(nu):
                f_nu[nu_idx, col] = t_nu[nu_idx, col]
                for row in range(nu_idx - 1, -1, -1):
                    angle = angles_nu[row]
                    sin = np.sin(angle)
                    cos = np.cos(angle)
                    new_t_nu[row, col] = cos * t_nu[row, col] - sin * f_nu[row + 1, col]
                    f_nu[row, col] = sin * t_nu[row, col] + cos * f_nu[row + 1, col]
            t_nu = new_t_nu

    # for nu = 1 there are no angles (A^1 = id), we put it in for convenience and
    # completeness
    angles.append([])
    angles.reverse()
    return angles


# decomposition of the A^nu matrices into the R^nu_k matrices as in the paper
def get_matrices(
    angles: list[list[float]],
    transpose: bool = False,
) -> list[NDArray[np.float64]]:
    """the matrices are similarly reversed as the angles..."""
    if transpose:
        sign = -1
    else:
        sign = 1
    matrices = []
    n = len(angles)
    for nu in range(2, n + 1):
        nu_idx = nu - 1
        angles_nu = angles[nu_idx]
        r_matrices = []
        for mu in range(1, nu):
            mu_idx = mu - 1
            theta = angles_nu[mu_idx]
            r = np.identity(n, dtype=np.float64)
            r[mu_idx, mu_idx] = np.cos(theta)
            r[mu_idx, nu_idx] = np.sin(theta) * sign
            r[nu_idx, mu_idx] = -np.sin(theta) * sign
            r[nu_idx, nu_idx] = np.cos(theta)
            r_matrices.append(r)
        matrices.append(r_matrices)
    return matrices


def reconstruct_matrix(
    matrices: list[NDArray[np.float64]], transposed: bool = False
) -> NDArray[np.float64]:
    n = -1
    for ms in matrices:
        if len(ms) > 0:
            n = ms[0].shape[0]
            break
    assert n != -1
    matrix = np.identity(n, dtype=np.float64)
    if transposed:
        for rs in matrices:
            for r in rs:
                matrix = matrix @ r
    else:
        for rs in matrices:
            for r in rs:
                matrix = r @ matrix
    return matrix


def final_check(angles: list[Tuple[int, int, float]], km: NDArray[np.float64]) -> float:
    n = km.shape[0]
    reconstructed = np.identity(n, dtype=np.float64)
    for i, j, angle in angles:
        r = np.identity(n, dtype=np.float64)
        i_idx = i - 1
        j_idx = j - 1
        r[i_idx, i_idx] = np.cos(angle)
        r[i_idx, j_idx] = np.sin(angle)
        r[j_idx, i_idx] = -np.sin(angle)
        r[j_idx, j_idx] = np.cos(angle)
        reconstructed = r @ reconstructed
    norm = np.sum(np.abs(km).flatten())
    err = np.sum(np.abs(km - reconstructed).flatten())
    rel = err / norm
    print(f"final decomposition error: {err} to norm {norm} (rel: {rel})")
    return rel


# that's basically what we will be calling
def checked_get_angles_and_indices(
    km: NDArray[np.float64],
    block_sizes: list[int],
) -> list[Tuple[int, int, np.float64]]:
    """
    return list of [(1, 2, theta_1^2), (1, 3, theta_1^3), (2, 3, theta_2^3), ...]
    such that km = ... r_1_3 cdot r_2_3 cdot r_1_2; where r_k_l is e^{4 i theta_k^l Y_k_l}
    with Y_k_l being the Pauli Y operator acting on the modes k and l.
    """
    blocks = np_helper.view_blocks(km, block_sizes)

    class Decomposed(Enum):
        SUCCESS = 1
        UNFILTERED_SUCCESS = 2
        FAILURE = 3

    def decompose_block(
        block: NDArray[np.float64], transpose: bool, offset: int
    ) -> Tuple[
        Decomposed, float, list[Tuple[int, int, np.float64]], NDArray[np.float64]
    ]:
        if transpose:
            angles = get_angles(block.T)
        else:
            angles = get_angles(block)
        matrices = get_matrices(angles, transpose=transpose)
        reconstructed = reconstruct_matrix(matrices, transposed=transpose)
        if not np.allclose(block, reconstructed):
            err = np.sum(np.abs(block - reconstructed).flatten())
            ret = []
            for nu, angles_nu in enumerate(angles[1:], start=2):
                for k, angle in enumerate(angles_nu, start=1):
                    ret.append((offset + k, offset + nu, angle))
            if transpose:
                ret.reverse()
                ret = [(i, j, -angle) for (i, j, angle) in ret]
            return (Decomposed.FAILURE, err, ret, reconstructed)
        # filter out the identities (which are caused by having multiple components in the
        # line graph as this gives rise to block-diagonal K matrices)
        filtered_matrices = []
        ret = []
        for nu, angles_nu in enumerate(angles[1:], start=2):
            filtered_matrices_nu = []
            for k, angle in enumerate(angles_nu, start=1):
                if not np.isclose(angle % (2 * np.pi), 0.0):
                    filtered_matrices_nu.append(matrices[nu - 2][k - 1])
                    ret.append((offset + k, offset + nu, angle))
            if not len(filtered_matrices_nu) == 0:
                filtered_matrices.append(filtered_matrices_nu)
        if not len(filtered_matrices) == 0:
            reconstructed = reconstruct_matrix(filtered_matrices, transposed=transpose)
            if not np.allclose(block, reconstructed):
                err = np.sum(np.abs(block - reconstructed).flatten())
                if transpose:
                    ret.reverse()
                    ret = [(i, j, -angle) for (i, j, angle) in ret]
                return (Decomposed.UNFILTERED_SUCCESS, err, ret, reconstructed)
        if transpose:
            ret.reverse()
            ret = [(i, j, -angle) for (i, j, angle) in ret]
        err = np.sum(np.abs(block - reconstructed).flatten())
        return (Decomposed.SUCCESS, err, ret, reconstructed)

    full_ret = []

    def try_decompose_block(
        block: NDArray[np.float64], offset: int, acceptance_err: float = 1e-5
    ) -> Tuple[Union[list[Tuple[int, int, np.float64]], None], NDArray[np.float64]]:
        stat, err, ret, rec = decompose_block(block, False, offset)
        norm = np.sum(np.abs(block).flatten())
        if stat == Decomposed.SUCCESS:
            # print(f"decomposition: non-transposed successful (rel err: {err/norm})")
            return ret, rec
        elif stat == Decomposed.UNFILTERED_SUCCESS:
            t_stat, t_err, t_ret, t_rec = decompose_block(block, True, offset)
            if t_stat == Decomposed.SUCCESS:
                # print(
                #     "decomposition: transposed succesful over non-transposed-unfiltered "
                #     f"(rel err: {t_err/norm})"
                # )
                return t_ret, t_rec
            elif t_stat == Decomposed.UNFILTERED_SUCCESS:
                if t_err < err:
                    # print(
                    #     "decomposition: transposed-unfiltered better than non-transposed "
                    #     f"(rel err: {t_err/norm})"
                    # )
                    return t_ret, t_rec
                else:
                    # print(
                    #     "decomposition: non-transposed-unfiltered better than transposed "
                    #     f"(rel err: {err/norm})"
                    # )
                    return ret, rec
            else:
                return ret, rec
        else:
            t_stat, t_err, t_ret, t_rec = decompose_block(block, True, offset)
            if t_stat == Decomposed.SUCCESS:
                # print(f"decomposition: transposed successful (rel err: {t_err/norm})")
                return t_ret, t_rec
            elif t_stat == Decomposed.UNFILTERED_SUCCESS:
                # print(
                #     "decomposition: transposed-unfiltered successful "
                #     f"(rel err: {t_err/norm})"
                # )
                return t_ret, t_rec
            else:
                if t_err < err:
                    # print(f"decomposition fail: transposed better than non-transposed")
                    ferr = t_err
                    fret = t_ret
                    frec = t_rec
                else:
                    # print(f"decomposition fail: non-transposed better than transposed")
                    ferr = err
                    fret = ret
                    frec = rec
                # close-to-zero issues have an error of 1e-5 (roughly) for the current
                # model, which is h = Heisenberg2D(3, 3, Interaction(RandomCoupling(-1, 1,
                # seed), 0, 0, 0, RandomCoupling(-1, 1, seed), 0, 0, 0, RandomCoupling(-1,
                # 1, seed),), 0, False)
                if ferr / norm < acceptance_err:
                    # print(
                    #     f"...but acceptable relative error ({ferr/norm}), continuing..."
                    # )
                    return fret, frec
                else:
                    # print(
                    #     "decomposition failed completely with best error "
                    #     f"{ferr} to norm {norm} (rel: {ferr/norm})"
                    # )
                    return None, frec

    for i, block in enumerate(blocks):
        offset = sum(block_sizes[:i])
        block_ret, _ = try_decompose_block(block, offset, acceptance_err=1e-7)
        if block_ret is None:
            # print("attempting close-to-zero fix by adding small values...")
            new_block = block.copy()
            for i in range(block.shape[0]):
                for j in range(block.shape[1]):
                    if np.abs(new_block[i, j]) < 1e-11:
                        new_block[i, j] = 1e-11
            new_ret, _ = try_decompose_block(new_block, offset, acceptance_err=1e-4)
            if new_ret is None:
                raise RuntimeError(
                    "decomposition failed completely even after hacky fix"
                )
            else:
                # print("...success after close-to-zero fix.")
                full_ret.extend(new_ret)
        else:
            full_ret.extend(block_ret)

    final_check(full_ret, km)
    return full_ret


def run():
    from scipy.stats import special_ortho_group

    dim = 50
    so_generator = special_ortho_group(dim=dim, seed=0)

    for i in range(100):
        print(i)
        km = so_generator.rvs()
        angles = get_angles(km)
        matrices = get_matrices(angles)
        check = reconstruct_matrix(matrices)
        assert np.allclose(km, check)
        # now do the same but transposed, which gives us the order as we have it in the
        # paper
        angles = get_angles(km.T)
        matrices = get_matrices(angles, transpose=True)
        check = reconstruct_matrix(matrices, transposed=True)
        assert np.allclose(km, check)


# # alternative method from the Hoffman paper
#
# # this first method from the paper is more sensitive to the close-to-zero-issues
# # as the equalisation of the sines and cosines fail fairly easily
# if nu > 2:
#     k = 1
#     k_idx = k - 1
#     arg = t_nu[k_idx, nu_idx]
#     if np.abs(arg) > 1.0:
#         assert np.isclose(np.abs(arg), 1.0)
#         arg = np.sign(arg) * 1.0
#     theta = np.arcsin(arg)
#     # note that np.arcsin returns values in [-pi/2, pi/2]
#     angles_nu.append(theta)
#     if np.abs(theta) == np.pi / 2:
#         for _ in range(2, nu):
#             angles_nu.append(0.0)
#         angles.append(angles_nu)
#         continue
# if nu > 3:
#     for k in range(2, nu - 1):
#         k_idx = k - 1
#         arg = t_nu[k_idx, nu_idx] / np.prod([np.cos(a) for a in angles_nu])
#         if np.abs(arg) > 1.0:
#             assert np.isclose(np.abs(arg), 1.0)
#             arg = np.sign(arg) * 1.0
#         theta = np.arcsin(arg)
#         if np.abs(theta) == np.pi / 2:
#             for _ in range(k + 1, nu):
#                 angles_nu.append(0.0)
#             break
#         angles_nu.append(theta)
# k = nu - 1
# k_idx = k - 1
# assert k == len(angles_nu) + 1
# arg_s = t_nu[k_idx, nu_idx] / np.prod([np.cos(a) for a in angles_nu])
# if np.abs(arg_s) > 1.0:
#     assert np.isclose(np.abs(arg_s), 1.0)
#     arg_s = np.sign(arg_s) * 1.0
# theta_s = np.arcsin(arg_s)
# arg_c = t_nu[k_idx + 1, nu_idx] / np.prod([np.cos(a) for a in angles_nu])
# if np.abs(arg_c) > 1.0:
#     assert np.isclose(np.abs(arg_c), 1.0)
#     arg_c = np.sign(arg_c) * 1.0
# theta_c = np.arccos(arg_c)
# # np.arcsin returns values in [-pi/2, pi/2] and np.arccos returns values in [0,
# # pi]; we need to adjust them to be the same and in [-pi, pi]:
# if np.isclose(theta_s, 0.0):
#     if not np.isclose(theta_c, 0.0):
#         theta_c = np.pi - theta_c
#     assert np.isclose(theta_c, 0.0)
# elif theta_s > 0.0:
#     if not np.isclose(theta_s, theta_c):
#         theta_s = np.pi - theta_s
#     assert np.isclose(theta_s, theta_c)
# elif theta_s < 0.0:
#     theta_c = -theta_c
#     if not np.isclose(theta_s, theta_c):
#         theta_s = -np.pi - theta_s
#     assert np.isclose(theta_s, theta_c)
# else:
#     raise RuntimeError("unreachable")
# theta = theta_s
# angles_nu.append(theta)
# angles.append(angles_nu)
