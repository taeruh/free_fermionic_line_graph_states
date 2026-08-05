import numpy as np
from hamiltonian import Hamiltonian
from paulis import Pauli

from models import Coupling


class Interaction:
    def __init__(
        self,
        xx: Coupling,
        xy: Coupling,
        xz: Coupling,
        yx: Coupling,
        yy: Coupling,
        yz: Coupling,
        zx: Coupling,
        zy: Coupling,
        zz: Coupling,
    ):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yx = yx
        self.yy = yy
        self.yz = yz
        self.zx = zx
        self.zy = zy
        self.zz = zz

    def __str__(self):
        return f"Interaction(xx={self.xx}, xy={self.xy}, xz={self.xz}, yx={self.yx}, yy={self.yy}, yz={self.yz}, zx={self.zx}, zy={self.zy}, zz={self.zz})"


class Heisenberg2D:
    """Square lattice of m x n spins"""

    def __init__(
        self,
        m: int,
        n: int,
        interaction: Interaction,
        hz: float,
        periodic: bool = False,
    ):
        self.m = m
        self.n = n
        self.interaction = interaction
        self.hz = hz
        if periodic:
            m_interaction_end = m
            n_interction_end = n
        else:
            m_interaction_end = m - 1
            n_interction_end = n - 1
        weights = []
        ops = []
        weights_generator: list[Coupling] = []
        ops_generator = []
        # if interaction.xx != 0 or isinstance(interaction.xx, RandomCoupling):
        #     weights_generator.append(interaction.xx)
        if not interaction.xx.is_zero():
            #                      z1     x1     z2    x2   phase
            weights_generator.append(interaction.xx)
            ops_generator.append((False, True, False, True, 0))
        if not interaction.xy.is_zero():
            weights_generator.append(interaction.xy)
            ops_generator.append((False, True, True, True, 3))
        if not interaction.xz.is_zero():
            weights_generator.append(interaction.xz)
            ops_generator.append((False, True, True, False, 0))
        if not interaction.yx.is_zero():
            weights_generator.append(interaction.yx)
            ops_generator.append((True, True, False, True, 3))
        if not interaction.yy.is_zero():
            weights_generator.append(interaction.yy)
            ops_generator.append((True, True, True, True, 2))
        if not interaction.yz.is_zero():
            weights_generator.append(interaction.yz)
            ops_generator.append((True, True, True, False, 3))
        if not interaction.zx.is_zero():
            weights_generator.append(interaction.zx)
            ops_generator.append((True, False, False, True, 0))
        if not interaction.zy.is_zero():
            weights_generator.append(interaction.zy)
            ops_generator.append((True, False, True, True, 3))
        if not interaction.zz.is_zero():
            weights_generator.append(interaction.zz)
            ops_generator.append((True, False, True, False, 0))
        if m == 1 and n == 1:
            pass
        if m == 1:
            for j in range(n_interction_end):
                jnext = (j + 1) % n
                for wx, op in zip(weights_generator, ops_generator):
                    z = np.zeros(m * n, dtype=bool)
                    x = np.zeros(m * n, dtype=bool)
                    z[j] = op[0]
                    x[j] = op[1]
                    z[jnext] = op[2]
                    x[jnext] = op[3]
                    weights.append(wx.sample())
                    ops.append(Pauli(m * n, z, x, op[4]))
        if n == 1:
            for i in range(m_interaction_end):
                inext = (i + 1) % m
                for wx, op in zip(weights_generator, ops_generator):
                    z = np.zeros(m * n, dtype=bool)
                    x = np.zeros(m * n, dtype=bool)
                    z[i] = op[0]
                    x[i] = op[1]
                    z[inext] = op[2]
                    x[inext] = op[3]
                    weights.append(wx.sample())
                    ops.append(Pauli(m * n, z, x, op[4]))
        else:
            for i in range(m_interaction_end):
                inext = (i + 1) % m
                for j in range(n_interction_end):
                    jnext = (j + 1) % n
                    for wx, op in zip(weights_generator, ops_generator):
                        z = np.zeros(m * n, dtype=bool)
                        x = np.zeros(m * n, dtype=bool)
                        z[i * n + j] = op[0]
                        x[i * n + j] = op[1]
                        v_z = z.copy()
                        v_x = x.copy()
                        v_z[i * n + jnext] = op[2]
                        v_x[i * n + jnext] = op[3]
                        weights.append(wx.sample())
                        ops.append(Pauli(m * n, v_z, v_x, op[4]))
                        z[inext * n + j] = op[2]
                        x[inext * n + j] = op[3]
                        weights.append(wx.sample())
                        ops.append(Pauli(m * n, z, x, op[4]))
        if hz != 0:
            for i in range(m):
                for j in range(n):
                    z = np.zeros(m * n, dtype=bool)
                    x = np.zeros(m * n, dtype=bool)
                    z[i * n + j] = True
                    weights.append(hz)
                    ops.append(Pauli(m * n, z, x, 0))

        self.hamiltonian = Hamiltonian(weights, ops)
