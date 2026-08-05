# Whyyyyy is there no (good) standard library for that kind of stuff?!?!? (numpy, scipy,
# ... have functions seem to be doing that, but even when the dtype is specified as bool
# it actually deals with it internally as integers, which is not the same, i.e., True +
# True -> 1 + 1 = 2 = True in bool (everything but 0 gets converted to True), but in F2 we
# want True + True = 0); sympy has a method to get the rank and one can specify a
# `iszeroval` function, but the implementation is much more complicated than what we are
# doing here and it is not entirely clear whether there are some catches to the
# `iszeroval` function) EDIT: sagemath has it (matrix(GF(2), <data>)! However, it sucks:
# it uses symbolic variables and it is super annoying to always convert ints/bools to
# symbolic variables and back (also, the F2 matrix implementation is missing some basic
# functionality, e.g., like `insert_row` (other sagemath matrices have it), and so to
# append to the matrix one has to deconstruct it and construct it again...)

import numpy as np

from numpy.typing import NDArray
from typing import Tuple


class Matrix:
    def __init__(self, data: NDArray[np.bool]):
        assert data.ndim <= 2
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.dtype != np.bool:
            print("warning: matrix data is not of type np.bool; converting to np.bool")
            self._data = data.astype(np.bool)
        else:
            self._data = data
        self._m, self._n = self._data.shape

    @classmethod
    def from_rows(cls, rows: list["Vector"]) -> "Matrix":
        if not rows:
            raise ValueError("cannot create matrix from empty list of rows")
        n = rows[0].len
        for r in rows:
            assert r.len == n
        data = np.vstack([r.data for r in rows])
        return cls(data)

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    @property
    def data(self) -> NDArray[np.bool]:
        return self._data

    def clone(self) -> "Matrix":
        return Matrix(self._data.copy())

    def as_int(self) -> NDArray[np.int_]:
        return self._data.astype(np.int_)

    def __repr__(self) -> str:
        return str(self.as_int())

    def transpose(self) -> "Matrix":
        return Matrix(self._data.T)

    def append_row(self, row: "Vector") -> None:
        assert row.len == self._n
        self._data = np.vstack([self._data, row.data])
        self._m += 1

    # TODO: have someone else check the logic here
    def into_step_form(self) -> Tuple[int, list[int]]:
        """Convert the matrix into upper step form (in-place) and return its rank and the
        column steps."""
        matrix = self._data
        rank = 0
        col_steps = []
        for col in range(self._n):
            act_row = None
            for row in range(rank, self._m):
                if matrix[row, col] == True:
                    act_row = row
                    break

            if act_row is not None:
                col_steps.append(col)
                if act_row != rank:
                    matrix[rank], matrix[act_row] = (
                        matrix[act_row].copy(),
                        matrix[rank].copy(),
                    )

                for row in range(rank + 1, self._m):
                    if matrix[row, col] == True:
                        matrix[row] ^= matrix[rank]

                rank += 1
                if rank == self._m:
                    break

        return rank, col_steps

    # TODO: have someone else check the logic here
    def step_form_into_reduced_step_form(
        self, rank: int, column_steps: list[int]
    ) -> None:
        """Assume the matrix is in step form..."""
        matrix = self._data
        for act_row in range(rank - 1, -1, -1):
            act_col = column_steps[act_row]
            for row in range(act_row - 1, -1, -1):
                if matrix[row, act_col] == True:
                    matrix[row] ^= matrix[act_row]

    def into_reduced_step_form(self) -> Tuple[int, list[int]]:
        rank, col_steps = self.into_step_form()
        self.step_form_into_reduced_step_form(rank, col_steps)
        return rank, col_steps

    def get_kernel_from_reduced_step_form(
        self, rank: int, column_steps: list[int]
    ) -> list["Vector"]:
        """Assume the matrix is in reduced step form and return a basis for its kernel."""
        basis_vectors = []
        free_vars = [i for i in range(self._n) if i not in column_steps]
        for free_var in free_vars:
            vec = np.zeros(self._n, dtype=np.bool)
            vec[free_var] = True
            for row in range(rank):
                col_step = column_steps[row]
                if self._data[row, free_var] == True:
                    vec[col_step] = True
            basis_vectors.append(Vector(vec))
        return basis_vectors

    def get_kernel(self) -> list["Vector"]:
        """Convert the matrix into reduced step form and return a basis for its kernel."""
        rank, col_steps = self.into_reduced_step_form()
        return self.get_kernel_from_reduced_step_form(rank, col_steps)

    def multiply(self, other: "Matrix") -> "Matrix":
        assert self._n == other._m
        result = np.zeros((self._m, other._n), dtype=np.bool)
        for i in range(self._m):
            for j in range(other._n):
                for k in range(self._n):
                    result[i, j] ^= self._data[i, k] & other._data[k, j]
        return Matrix(result)

    def left_multiply_row_vector(self, vec: "Vector") -> "Matrix":
        left = Row(vec.data)
        return left.multiply(self)

    def right_multiply_column_vector(self, vec: "Vector") -> "Matrix":
        right = Column(vec.data)
        return self.multiply(right)


class Vector:
    def __init__(self, data: NDArray[np.bool]):
        assert data.ndim == 1
        if data.dtype != np.bool:
            print("warning: vector data is not of type np.bool; converting to np.bool")
            self.data = data.astype(np.bool)
        else:
            self.data = data
        self.len = self.data.shape[0]

    def clone(self) -> "Vector":
        return Vector(self.data.copy())

    def __repr__(self) -> str:
        return str(self.data.astype(np.int_))

    def add(self, other: "Vector") -> "Vector":
        assert self.len == other.len
        return Vector(self.data ^ other.data)


class Row(Matrix):
    def __init__(self, data: NDArray[np.bool]):
        assert data.ndim == 1
        super().__init__(data.reshape(1, -1))


class Column(Matrix):
    def __init__(self, data: NDArray[np.bool]):
        assert data.ndim == 1
        super().__init__(data.reshape(-1, 1))


def vector(v: list[int]) -> Vector:
    return Vector(np.array(v, dtype=np.bool))


def max_independent_set(vecs: list[Vector]) -> list[Vector]:
    # there might be a more efficient way to do this more directly (maybe via the
    # "Replacement theorem of Steinitz"?), but this is simple and works (because the
    # dimension of vectorspaces is well-defined)
    if not vecs:
        return []

    ret = [vecs[0]]
    matrix = Matrix.from_rows([vecs[0]])
    rank = 1
    for e in vecs[1:]:
        test_matrix = matrix.clone()
        test_matrix.append_row(e)
        test_rank, _ = test_matrix.into_step_form()
        if test_rank > rank:
            ret.append(e)
            matrix = test_matrix
            rank += 1

    return ret


def linear_decomposition(basis: list[Vector], vec: Vector) -> list[int] | None:
    """
    assume basis is linearly independent and vec is in the span of basis
    return None if this is not the case and the decomposition fails
    """
    # what we do here only works because of the above assumption; more specifically: let A
    # be the matrix with the columns being the basis vectors and let b be the vector; we
    # want to solve Ax = b; in general, this is done by transforming the matrix (A b) into
    # (A' b') such that A' is in reduced step form (not the full (A' b')); we know that the
    # columns of A are linearly independent, so when doing gaussian elimination on the
    # whole (A' b') (this is what the method below does), up to the
    # second last column in (A b) we will have (A'' b'') where A'' is upper triangular;
    # now we know that for the rows below the rank of A'', the corresponding entry in b''
    # must be zero since we assumed that b is in the span of the columns of A; thus, the
    # first step of the gaussian elimination is finished an (A'' b'') is the reduced step
    # form; but now we just bring (A'' b'') into reduced step form by eliminating by
    # effecitively bringing A'' into reduced step form and we get (A''' b'''); we see that
    # effectively, we did the gaussian elimination on (A b) only guided by A which is
    # exactly what we wanted to do in the first place
    mat = Matrix.from_rows(basis + [vec]).transpose()
    _, _ = mat.into_reduced_step_form()
    decom = []
    n = len(basis)
    for i in range(n):
        if mat.data[i, n]:
            decom.append(i)
    # let's test whether this is correct
    test_vec = Vector(np.zeros(vec.len, dtype=np.bool))
    for i in decom:
        test_vec = test_vec.add(basis[i])
    if not np.array_equal(test_vec.data, vec.data):
        return None
    return decom
