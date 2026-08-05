import numpy as np

# import linalg.f2.matrix
from linalg.f2 import matrix
from linalg.f2.matrix import Matrix, Vector

from numpy.typing import NDArray


class SymplecticVector(Vector):
    def __init__(self, data: NDArray[np.bool]):
        super().__init__(data)
        assert self.len % 2 == 0
        self.n = self.len // 2
        self.u = self.data[: self.n]
        self.l = self.data[self.n : 2 * self.n]

    @classmethod
    def from_vector(cls, v: Vector) -> "SymplecticVector":
        return SymplecticVector(v.data)

    def __repr__(self):
        return f"({self.u.astype(np.int_)},{self.l.astype(np.int_)}; {self.n})"

    def symplectic_inner_product(self, other: "SymplecticVector") -> bool:
        assert self.n == other.n
        ip = False
        for i in range(self.n):
            ip ^= (self.l[i] & other.u[i]) ^ (self.u[i] & other.l[i])
        return ip


def get_standard_symplectic_form(n: int) -> Matrix:
    """Returns te standard symplectic form Omega over F_2 for 2n-dimensional
    vectorspaces."""
    omega = np.zeros((2 * n, 2 * n), dtype=np.bool)
    for i in range(n):
        omega[i, n + i] = True
        omega[n + i, i] = True
    return Matrix(omega)


def stabilisers_all_commuting(vecs: list[SymplecticVector]) -> bool:
    if len(vecs) == 0:
        return True
    identity = np.zeros(vecs[0].n, dtype=int)
    for i in range(len(vecs)):
        # don't allow negative identity, and also not positive identity as this one gives
        # no information and should not be in the list
        if vecs[i].data.tolist() == identity.tolist():
            print("found identity in stabiliser")
            return False
        for j in range(i + 1, len(vecs)):
            if vecs[i].n != vecs[j].n or vecs[i].symplectic_inner_product(vecs[j]):
                return False
    return True


# TODO: have someone else check the logic here
def extend_to_full_isotropic_set(
    vecs: list[SymplecticVector],
) -> list[SymplecticVector]:
    """
    this assumes that `vecs` is already isotropic and linearly independent!!!
    changes `vecs` (i.e., appends to the list)
    """
    if len(vecs) == 0:
        raise ValueError("cannot extend empty set of Paulis (don't know n)")

    ret = vecs.copy()

    n = vecs[0].n
    m = len(vecs)

    omega = get_standard_symplectic_form(n)
    mat = Matrix.from_rows(vecs)  # pyright: ignore

    while m < n:
        ker = (mat.multiply(omega)).get_kernel()
        if len(ker) == 0:
            raise ValueError(
                f"bug: something is deeply wrong, no kernel but m < n ({m} < {n})"
            )

        # ker is a full basis and its span includes ret, so we extend ret by ker and then
        # find a maximal independent subset (the method automatically includes ret as we
        # assumed that ret is independent); the extra elements in that set must contain at
        # least one new element, so we can add this one definitely to ret, and for the
        # remaining elements we can add them if they commute with all the ones we added in
        # the previous adding step

        optimistic_try = ret.copy()
        for e in ker:
            optimistic_try.append(SymplecticVector.from_vector(e))
        optimistic_try: list[SymplecticVector] = matrix.max_independent_set(
            optimistic_try  # pyright: ignore
        )
        optimistic_m = len(optimistic_try)
        assert optimistic_m >= m

        potential_vecs = optimistic_try[m:]
        new_vecs = [potential_vecs[0]]
        for candidate in potential_vecs[1:]:
            commutes_with_all = True
            for p in new_vecs:
                if candidate.symplectic_inner_product(p) != 0:
                    commutes_with_all = False
                    break
            if commutes_with_all:
                new_vecs.append(candidate)
        assert len(new_vecs) > 0

        ret.extend(new_vecs)
        m = len(ret)

    return ret


# TODO: have someone else check the logic here
def get_full_hyperbolic_partners(
    vecs: list[SymplecticVector],
) -> list[SymplecticVector]:
    """
    given a full isotropic linearly independent set; return the hyperbolic partners, i.e.,
    the set of vectors that together with the input set form a symplectic basis
    """
    if len(vecs) == 0:
        raise ValueError("cannot extend empty set of Paulis (don't know n)")

    ret = []

    n = vecs[0].n
    assert len(vecs) == n

    omega = get_standard_symplectic_form(n)

    for i in range(n):
        mat = Matrix.from_rows(vecs[:i] + vecs[i + 1 :] + ret)  # pyright: ignore
        ker = (mat.multiply(omega)).get_kernel()
        # basis = vecs + ret  # this here is what we want, but the following is sufficient
        # (can be proven via induction (might require some more work with symplectic
        # spaces)); if using `basis = vecs + ret`, change the `if` condition below to `==
        # n+i+1` instead of `== n-i+1`
        basis = vecs[i:]
        for e in ker:
            test_matrix = Matrix.from_rows(basis.copy())  # pyright: ignore
            test_matrix.append_row(e)
            test_rank, _ = test_matrix.into_step_form()
            if test_rank == n - i + 1:
                ret.append(SymplecticVector.from_vector(e))
                assert vecs[i].symplectic_inner_product(ret[-1]) == 1
                break

    check_basis = vecs + ret
    check_matrix = Matrix.from_rows(check_basis.copy())  # pyright: ignore
    check_rank, _ = check_matrix.into_step_form()
    print(len(vecs), check_rank)
    assert check_rank == 2 * n
    assert len(ret) == n
    return ret
