'''
Compiling Shor's Algorithm.

@Author: Marcus Edwards
@Date: 2025-02-20
'''

import random
import numpy as np
import pennylane as qml
from pennylane import numpy as np
from functools import partial


INPUT_QUBITS = 10
OUTPUT_QUBITS = 5


dev = qml.device('default.qubit', wires=3, shots=None)


# Part 1. Helper functions


def get_gcd(l, N):
    """Find the GCD of two numbers.

    Args:
        l (int): First number.
        N (int): Second number.

    Returns:
        int: the GCD.
    """
    r = l % N

    while r:
        l = N
        N = r
        r = l % N

    return N


def is_coprime(l, N):
    """Determine if two numbers are coprime.

    Args:
        l (int): First number to check if is coprime with the other.
        N (int): Second number to check if is coprime with the other.

    Returns:
        bool: True if they are coprime numbers, False otherwise.
    """

    return get_gcd(l, N) == 1


def is_odd(r):
    """Determine if a number is odd.

    Args:
        r (int): Integer to check if is an odd number.

    Returns:
        bool: True if it is odd, False otherwise.
    """

    return r % 2 == 1


def is_not_one(x, N):
    """Determine if x is not +- 1 modulo N.

    Args:
        N (int): Modulus of the equivalence.
        x (int): Integer to check if it is different from +-1 modulo N.

    Returns:
        bool: True if it is different, False otherwise.
    """

    return x % N != N - 1 and x % N != 1


def shor(N, l=None, e=10**-17):
    """Return the factorization of a given integer.

    Args:
       N (int): integer we want to factorize.
       l (int): Optional choice of integer coprime to N.
       e (float): Optional error which the circuit is compiled to.

    Returns:
        array[int]: [p,q], the prime factors of N.
    """

    if l is None:
        l = random.randint(2, N - 2)
    if not is_coprime(l, N):
        p = get_gcd(l, N)
        q = N / p
    else:
        Us = get_matrix_l_mod_N(l, N)
        U = make_exponentiation_matrix_controlled(Us)
        r = get_period(U, e)
        if is_odd(r):
            [p, q] = shor(N)
        else:
            x = (l ** (r / 2)) % N
            if not is_not_one(x):
                [p, q] = shor(N)
            else:
                p = get_gcd(x - 1, N)
                q = get_gcd(x + 1, N)
    return [p, q]


# Part 2. Compiling CNOT circuits.


def make_exponentiation_matrix_controlled(Us):
    """
    Injects the control parameter into the modular exponentation circuits.
    :param Us: The modular exponentation circuits.
    :return: A function that calls all the controlled exponentation circuits.
    """
    def modular_exponentiation_matrix_controlled():
        index = 0
        for U in Us:
            U(c=INPUT_QUBITS - index - 1)
            index += 1

    return modular_exponentiation_matrix_controlled


def get_matrix_l_mod_N(l, N):
    """
    Synthesizes the U_{N,l} operator's matrix in terms of CNOTs.

    :param l: the integer coprime to N which we are finding the period of.
    :param N: the modulus being factored.
    :return: The matrix U_{N,l}.
    """
    num_qubits = 3
    i = 0
    unitaries = []
    while i < 2*num_qubits - 1:
        unitary = get_controlled_modular_multiplication_unitary(l, N, i)
        unitaries.append(unitary)
        i += 1
    return unitaries


def get_controlled_modular_multiplication_unitary(l, N, i):
    """
    Returns an in-place controlled modular multiplication unitary for power i.
    :param l: The integer coprime to N.
    :param N: The secret N.
    :param i: The power of l.
    :return:
    """
    truth_table = {}

    for x in range(2**OUTPUT_QUBITS):
        truth_table[format(x, '#07b').split('b')[1]] = format((x*(l**(2**i)) % N), '#07b').split('b')[1]

    U = []

    for op_row in range(OUTPUT_QUBITS):
        b = []
        a = []
        for input in truth_table.keys():
            b.append(int(truth_table[input][op_row]))
            a.append([int(input[input_entry]) for input_entry in range(OUTPUT_QUBITS)])
        row = np.linalg.lstsq(np.array(a), np.array(b))
        U.append(row)

    return partial(CNOT_synth, A=np.array(U), n=len(row), m=2)


def CNOT_synth(A, n, m, c):
    """
    Performs the CNOT circuit synthesis using the efficient approach in
    [1] K. N. Patel, I. L. Markov, and J. P. Hayes, “Efficient Synthesis
    of Linear Reversible Circuits,” Feb. 03, 2003, arXiv: arXiv:quant-ph/0302002.
    doi: 10.48550/arXiv.quant-ph/0302002.

    :param A: The initial matrix to row-reduce.
    :param n: The dimension of the square matrix.
    :param m: The size of the partitions of the square matrix.
    :param c: The qubit that controls this operation.
    :return: The circuit.
    """
    # synthesize upper and lower triangular parts
    [A, circuit_lower] = lwr_CNOT_synth(A, n, m)
    A = np.transpose(A)
    [A, circuit_upper] = lwr_CNOT_synth(A, n, m)

    # switch control / target of CNOT in upper part.
    for i in range(len(circuit_upper)):
        temp = circuit_upper[i][1]
        circuit_upper[i][1] = circuit_lower[i][0]
        circuit_lower[i][0] = temp

    # combine upper, lower parts
    circuit = circuit_upper + circuit_lower

    # convert to pennylane circuit
    for j in range(len(circuit)):
        qml.MultiControlledX((c, *circuit[j]))


def lwr_CNOT_synth(A, n, m):
    """
    Helper function for the efficient CNOT circuit synthesis which synthesizes
    the lower triangular part.

    :param A: The square matrix to row-reduce.
    :param n: The size of the matrix.
    :param m: The size of the partitions of the square matrix.
    :return: The row-reduced matrix and the circuit.
    """
    circuit = []
    for sec in range(1, int(np.ceil(n/m))):  # iterate over column sections
        patt = dict()
        # remove duplicate sub-rows in section
        for i in range(2**m):
            patt[i] = -1
        for row_ind in range((sec-1)*m, n):
            sub_row_patt = A[row_ind, (sec-1)*m: sec*m-1]
            if patt[sub_row_patt] == -1:
                patt[sub_row_patt] = row_ind
            else:
                A[row_ind, :] += A[patt[sub_row_patt], :]
                circuit = [(patt[sub_row_patt], row_ind)] + circuit

        # use Gaussian elimination for remaining entries in column section
        for col_ind in range((sec-1)*m, sec*m-1):
            # check for 1 on diagonal
            diag_one = (A[col_ind, col_ind] == 0)
            # remove ones in rows below col_ind
            for row_ind in range(col_ind+1, n):
                if A[row_ind, col_ind] == 1:
                    if not diag_one:
                        A[col_ind, :] += A[row_ind, :]
                        circuit = [(row_ind, col_ind)] + circuit
                        diag_one = 1
                    A[row_ind, :] += A[col_ind, :]
                    circuit = [(col_ind, row_ind)] + circuit

        return [A, circuit]


shor_machine = qml.device('default.qubit', wires=6, shots=None)


def get_period(U, e):
    """
    Calls the quantum subroutine and calculates the period r from the measurement outcome histogram.
    :param U: The modular exponentiation operator.
    :param N: The number to factor.
    :return: The period r.
    """
    # measure y_m = m (2^3 / r) with high probability
    probs = shor_circuit(U)

    # draw Clifford + T circuit for error e
    qml.draw(qml.clifford_t_decomposition(shor_circuit(U), e))

    # Calculate the period
    ys = []
    index = 0

    for prob in probs:
        if not np.isclose(prob, 0.):
            ys.append(index)
        index += 1

    rs = list(map(lambda m_y: (2**3 / (m_y[1] / m_y[0])), enumerate(ys)))

    for r in rs:
        assert np.isclose(r, np.sum(rs) / len(rs))

    return r


@qml.qnode(shor_machine)
def shor_circuit(U):
    """
    Computes the minimum r such that U^r |1> = |1>.
    :param U: The unitary that encodes the function l^x (mod N).
    :param N: The modulus, or public key, N.
    :return: The period.
    """

    # put input register in superposition
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)

    # apply modular exponentiation function
    U()

    # measure output register
    qml.measure(3)
    qml.measure(4)
    qml.measure(5)

    # apply QFT to input register
    qml.QFT(wires=[0, 1, 2, 3, 4])

    # measure input register
    return qml.probs(wires=[0, 1, 2])


if __name__ == '__main__':
    shor(32, 3, 10**-7)
