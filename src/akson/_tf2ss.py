import sympy as sp
import sympy.matrices.normalforms


def frac_matrix_to_right_mfd(M):
    """! Represent M as Matrix Fraction Description (MFD)

    @param M Matrix of rational functions

    @return N MFD numerator
    @return D MFD denominator
    """
    n = M.shape[0]
    m = M.shape[1]

    numerators = sp.zeros(n, m)
    denominators = sp.zeros(n, m)
    for i in range(n):
        for j in range(m):
            f_n, f_d = sp.fraction(M[i, j])
            numerators[i, j] = f_n
            denominators[i, j] = f_d

    col_prefix_polys = []
    col_suffix_polys = []
    for j in range(m):
        curr_prefix_polys = [denominators[0, j]]
        curr_suffix_polys = [denominators[n - 1, j]]
        for i in range(1, n):
            curr_prefix_polys.append(curr_prefix_polys[-1] * denominators[i, j])
            curr_suffix_polys.append(curr_suffix_polys[-1] * denominators[n - 1 - i, j])
        col_prefix_polys.append(curr_prefix_polys)
        col_suffix_polys.append([p for p in reversed(curr_suffix_polys)])

    D = sp.diag(*[suffix_polys[0] for suffix_polys in col_suffix_polys])
    N = sp.zeros(n, m)
    for j in range(m):
        for i in range(n):
            prefix = col_prefix_polys[j][i - 1] if i > 0 else 1
            suffix = col_suffix_polys[j][i + 1] if i < n - 1 else 1
            N[i, j] = numerators[i, j] * prefix * suffix
    return N, D


def get_column_degrees(M, s):
    """! Calculates column degrees of M

    @param M Matrix of polynomials
    @param s Indeterminate variable of polynomials in M

    @return column_degrees Degrees of columns of M
    """
    n = M.shape[0]
    m = M.shape[1]
    column_degrees = [
        max(sp.Poly(M[i, j], s).degree() for i in range(n)) for j in range(m)
    ]
    return column_degrees


def construct_D_hc(D, column_degrees, s):
    """! Constructs matrix of coefficient of each columns highest polynomial terms coefficients

    @param D Matrix of polynomials
    @param s Indeterminate variable of polynomials in M
    @param column_degrees Degrees of columns of M

    @return D_hc Matirx of coefficients of each columns highest polynomial terms coefficients
    """
    D_hc = sp.zeros(D.shape[0], D.shape[1])
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            poly = sp.Poly(D[i, j], s)
            if poly.degree() == column_degrees[j]:
                D_hc[i, j] = poly.coeffs()[0]
    return D_hc


def right_mfd_to_right_coprime(N, D, s):
    """!Computes right coprime form of a right MFD

    @return N MFD numerator
    @return D MFD denominator

    @return N_trimmed Trimmed version of N
    @return D_trimmed Trimmed version of D
    """
    n_N = N.shape[0]
    S, U, V = sp.matrices.normalforms.smith_normal_decomp(
        N.col_join(D), domain=sp.QQ[s]
    )
    U_inv = U.inv()
    new_m = S.rank()
    return U_inv[:n_N, :new_m], U_inv[n_N:, :new_m]


def column_reduce_right_mfd(N, D, s):
    """! Transforms MFD into column reduced form

    @return N MFD numerator
    @return D MFD denominator
    @param s Indeterminate variable of polynomials in M

    @return N_reduced Reduced version of N
    @return D_reduced Reduced version of D
    """
    N = N.copy()
    D = D.copy()

    n = D.shape[0]
    m = D.shape[1]

    # Validate that D is a correct MFD matrix
    assert D.shape[0] == m and D.shape[1] == m
    assert D.rank() == m

    # Necessary for MFD to be column-reducable
    assert m <= n

    while True:
        if D.rank() < D.shape[1]:
            raise ValueError("Bad D matrix")

        column_degrees = get_column_degrees(D, s)
        D_hc = construct_D_hc(D, column_degrees, s)

        # Check if D is already column reduced
        if D_hc.rank() == D_hc.shape[1]:
            return N, D

        # Zero one of the maximal terms
        nullspace_vector = D_hc.nullspace()[0]
        active_indices = [
            j for j in range(len(nullspace_vector)) if nullspace_vector[j] != 0
        ]
        cj = max(active_indices, key=lambda j: column_degrees[j])
        V = sp.eye(n)
        for i in active_indices:
            V[i, cj] = nullspace_vector[i] * (
                s ** (column_degrees[cj] - column_degrees[i])
            )
        D = (D @ V).expand()
        N = (N @ V).expand()


def extract_coeff(poly, deg):
    """! Extract coefficient of degree `deg` from polynomial `poly`

    @param poly Polynomial
    @param deg Target coefficient degree

    @return coefficient Coefficient of degree `deg` from polynomial `poly`
    """
    if poly.degree() >= deg:
        return poly.all_coeffs()[-1 - deg]
    else:
        return 0


def polynomial_matrix_to_lc(M, column_degrees, s):
    """! Construct matrix with columns corresponding to polynomial terms of consecutive columns of `M`

    @param M polynomial matrix
    @param column_degrees Matrix `M` column degrees
    @param s Indeterminate variable of polynomials in M

    @return M_lc Matrix with columns corresponding to polynomial terms of consecutive columns of `M`
    """
    columns = []
    for j in range(M.shape[1]):
        for deg in range(column_degrees[j]):
            columns.append(
                M[:, j].applyfunc(lambda e: extract_coeff(sp.Poly(e, s), deg))
            )
    return sp.Matrix.hstack(*columns)


def tf2ss_controller_form(H, s):
    """! Computes realization of transfer function matrix `H` as state space system in controller canonical form

    @param H Transfer funcion matrix
    @param s Indeterminate variable of polynomials in M

    @return Ac Matrix A of state space system in controller canonical form
    @return Bc Matrix B of state space system in controller canonical form
    @return Cc Matrix C of state space system in controller canonical form
    """
    # Validate that elements of H are rational functions
    n = H.shape[0]
    m = H.shape[1]
    for i in range(n):
        for j in range(m):
            num, denom = sp.fraction(H[i, j])
            num_poly = sp.Poly(num, s)
            denom_poly = sp.Poly(denom, s)
            assert num_poly.degree() < denom_poly.degree()

    # Represent H(s) as MFD
    N, D = frac_matrix_to_right_mfd(H)

    # Compute right coprime MFD
    N_trimmed, D_trimmed = right_mfd_to_right_coprime(N, D, s)

    if D_trimmed.shape[0] < D_trimmed.shape[1]:
        raise ValueError("System not controllable")

    # Compute the column reduced form
    N_reduced, D_reduced = column_reduce_right_mfd(N_trimmed, D_trimmed, s)

    # Realization
    column_degrees = get_column_degrees(D_reduced, s)
    D_hc = construct_D_hc(D_reduced, column_degrees, s)
    S_matrix = sp.diag(*[s**d for d in column_degrees])
    D_lc = polynomial_matrix_to_lc(D_reduced - D_hc @ S_matrix, column_degrees, s)
    N_lc = polynomial_matrix_to_lc(N_reduced, column_degrees, s)

    n = sum(column_degrees)
    m = m
    A0 = sp.zeros(n, n)
    base_i = 0
    for d in column_degrees:
        for offset in range(d - 1):
            A0[base_i + offset, base_i + offset + 1] = 1
        base_i += d

    B0 = sp.zeros(n, m)
    base_i = 0
    for j, d in enumerate(column_degrees):
        B0[base_i + d - 1, j] = 1
        base_i += d

    D_hc_inv = D_hc.inv()
    Ac = A0 - B0 @ D_hc_inv @ D_lc
    Bc = B0 @ D_hc_inv
    Cc = N_lc

    return Ac, Bc, Cc


def tf2ss_observer_form(H, s):
    """! Computes realization of transfer function matrix `H` as state space system in observer canonical form

    @param H Transfer funcion matrix
    @param s Indeterminate variable of polynomials in M

    @return Ao Matrix A of state space system in observer canonical form
    @return Bo Matrix B of state space system in observer canonical form
    @return Co Matrix C of state space system in observer canonical form
    """
    Ac, Bc, Cc = tf2ss_controller_form(H.T, s)
    Ao, Bo, Co = Ac.T, Cc.T, Bc.T

    return Ao, Bo, Co


def tf2ss(H, s):
    """! Computes realization of transfer function matrix `H` as state space system

    @param H Transfer funcion matrix
    @param s Indeterminate variable of polynomials in M

    @return Ac Matrix A of state space system
    @return Bc Matrix B of state space system
    @return Cc Matrix C of state space system
    """
    n = H.shape[0]
    m = H.shape[1]
    if n >= m:
        return tf2ss_controller_form(H, s)
    else:
        return tf2ss_observer_form(H, s)
