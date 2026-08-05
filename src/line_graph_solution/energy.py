def get(
    symmetry_chosen_sign_phases: list[int],
    lamda_chosen_sign_phases: list[int],
    lamdas: list[float],
    lamda_independent_indices: list[int],
    lamda_dependent_indices: list[int],
    splitted_dependent_decompositions: list[tuple[int, list[int], list[int]]],
    lamda_identites_indices: list[int],
    lamda_identity_phases: list[int],
):
    ret = 0
    for sign, idx in zip(lamda_chosen_sign_phases, lamda_independent_indices):
        ret += (-1) ** sign * lamdas[idx]
    for (phase, sym_decom_indices, lam_decom_indices), idx in zip(
        splitted_dependent_decompositions, lamda_dependent_indices
    ):
        assert phase % 2 == 0
        sign = (-1) ** (phase // 2)
        for decom_idx in sym_decom_indices:
            sign *= (-1) ** symmetry_chosen_sign_phases[decom_idx]
        for decom_idx in lam_decom_indices:
            sign *= (-1) ** lamda_chosen_sign_phases[decom_idx]
        ret += sign * lamdas[idx]
    for phase, idx in zip(lamda_identity_phases, lamda_identites_indices):
        assert phase % 2 == 0
        ret += (-1) ** (phase // 2) * lamdas[idx]
    return ret * 2
