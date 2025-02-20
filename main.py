'''
Compiling Shor's Algorithm.

@Author: Marcus Edwards
@Date: 2025-02-20
'''

import random
import numpy as np
import pennylane as qml
from pennylane import numpy as np


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


def shor(N):
    """Return the factorization of a given integer.

    Args:
       N (int): integer we want to factorize.

    Returns:
        array[int]: [p,q], the prime factors of N.
    """

    l = random.randint(2, N - 2)
    if not is_coprime(l, N):
        p = get_gcd(l, N)
        q = N / p
    else:
        U = get_matrix_a_mod_N(l, N)
        r = get_period(U, N)
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


def get_matrix_a_mod_N(l, N):
    """
    Synthesizes the U_{N,a} operator's matrix in terms of CNOTs.

    :param l: the integer coprime to N which we are finding the period of.
    :param N: the modulus being factored.
    :return: The matrix U_{N,a}.
    """
    pass


def CNOT_synth(A, n, m):
    """
    Performs the CNOT circuit synthesis using the efficient approach in
    [1] K. N. Patel, I. L. Markov, and J. P. Hayes, “Efficient Synthesis
    of Linear Reversible Circuits,” Feb. 03, 2003, arXiv: arXiv:quant-ph/0302002.
    doi: 10.48550/arXiv.quant-ph/0302002.

    :param A: The initial matrix to row-reduce.
    :param n: The dimension of the square matrix.
    :param m: The size of the partitions of the square matrix.
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

    # combine upper, lower parts and
    circuit = circuit_upper + circuit_lower


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
    for sec in range(1, np.ceil(n/m)):  # iterate over column sections
        patt = dict()
        # remove duplicate qub-rows in section
        for i in range(2**m):
            patt[i] = -1
        for row_ind in range((sec-1)*m, n):
            sub_row_patt = A[row_ind, (sec-1)*m : sec*m-1]  # TODO: what is this line supposed to index?
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


# Part 3. Phase estimation.


def fractional_binary_to_float(s):
    """ Helper function to expand fractional binary numbers as floats.

    :param s (string): A string in the form "0.xxxx" where the x are 0s and 1s.
    :return: The numerical value when converted from fractional binary to float.
    """

    assert '.' == s[1]
    assert s[0] == '0'

    for bit in s[2:]:
        assert bit in ('0', '1')

    bin = s.split('.')[-1]

    power_of_two = -1
    sum = 0
    for bit in bin:
        if bit == '1':
            sum += 2 ** power_of_two
        power_of_two -= 1

    return sum


def float_to_fractional_binary(x, max_bits=10):
    """ Helper function to turn a string to a binary representation.

    :param x (float): A numerical value between 0 < x < 1 with a decimal point.
    :param max_bits (int): The maximum number of bits in the expansion. For x that require
        fewer than max_bits for the expansion, terminate immediately.
    :return: A string that is the fractional binary representation, formatted as '0.bbbb'
        where there are max_bits b.
    """

    assert isinstance(x, float)
    assert 0 < x < 1

    s = '0.' + ''.join(['0' for i in range(max_bits)])

    index = 1
    while index < max_bits:
        if (2 ** (-1 * index)) <= (x - fractional_binary_to_float(s)):
            s = s[0: 2 + index] + '1' + s[2 + index + 1:]
        index += 1

    return s


def results_to_eigenvalue(results):
    """ Converts from the QPE probability histogram output to computed eigenvalue.

    :param results: The results from the QPE algorithm.
    :return: The eigenvalue computed.
    """

    for i, result in enumerate(results):
        if np.isclose(result, 1):
            break

    print(f"0.{str(bin(i)).split('b')[-1]}")
    flt = fractional_binary_to_float(f"0.{str(bin(i)).split('b')[-1]}")
    eigenvalue = complex(np.cos(2 * np.pi * flt), np.sin(2 * np.pi * flt))

    return eigenvalue


@qml.qnode(dev)
def qft_3():

    qml.Hadamard(wires=[0])
    qml.ControlledQubitUnitary(np.array([[1, 0], [0, 0-1j]]), control_wires=[1], wires=[0])  # Controlled-S gate
    qml.ControlledQubitUnitary(np.array([[1, 0], [0, complex(np.cos(np.pi / 4), -np.sin(np.pi / 4))]]), control_wires=[2], wires=[0])  # Controlled-T gate
    qml.Hadamard(wires=[1])
    qml.ControlledQubitUnitary(np.array([[1, 0], [0, 0 - 1j]]), control_wires=[2], wires=[1])  # Controlled-S gate
    qml.Hadamard(wires=[2])
    qml.SWAP(wires=[0, 2])

    return qml.probs(wires=[0, 1, 2])


@qml.qnode(dev)
def builtin_qft3():
    qml.QFT(wires=[0, 1, 2])

    return qml.probs(wires=[0, 1, 2])


dev2 = qml.device('default.qubit', wires=4, shots=None)


@qml.qnode(dev2)
def qpe(eigenvector):
    """ Quantum phase estimation on a single-qubit unitary with 3-bit precision.

    :return: The probability pf each of the basis states from qml.probs.
    """

    assert eigenvector in ('0', '1')

    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)

    # Initialize state in the eigenvector

    # The U in question has two eigenvectors: (1, 0) we denote |0> and (0, 1) we denote |1>
    if eigenvector == '1':
        qml.PauliX(3)

    # Perform controlled unitaries
    U = qml.QubitUnitary(np.array([[1, 0],
        [0, complex(np.cos(5*np.pi/4), np.sin(5*np.pi/4))]]), wires=[3])

    qml.ctrl(qml.pow(U, 2 ** 0), 2)
    qml.ctrl(qml.pow(U, 2 ** 1), 1)
    qml.ctrl(qml.pow(U, 2 ** 2), 0)

    # Now do the QFT backwards on the first three qubits
    qml.adjoint(qml.QFT(wires=[0, 1, 2]))

    return qml.probs(wires=[0, 1, 2])


@qml.qnode(dev2)
def builtin_qpe(eigenvalue):
    if eigenvalue == '1':
        qml.PauliX(3)

    U = qml.QubitUnitary(np.array([[1, 0],
                                   [0, complex(np.cos(5 * np.pi / 4), np.sin(5 * np.pi / 4))]]), wires=[3]).matrix()

    qml.QuantumPhaseEstimation(U, 3, [0, 1, 2])

    return qml.probs(wires=[0, 1, 2])


def get_period(U, N):
    """
    Computes the minimum r such that U^r |1> = |1>.
    :param U: The unitary that encodes the function a^m (mod N).
    :param N: The modulus, or public key, N.
    :return: The period.
    """
    pass


# Part 4. Convert to Clifford + T.


def convert_to_clifford_T(tape, epsilon):
    """
    Converts the provided circuit to a Clifford + T circuit with max allowed error epsilon.
    :param tape: The qnode to compile to Clifford + T.
    :param epsilon: The maximum allowed error epsilon.
    :return: The compiled circuit.
    """
    pass


if __name__ == '__main__':
    for val in qft_3():
        assert np.isclose(val, 0.125)  # should be a uniform superposition when acting on |0..0>

    assert np.allclose(qft_3(), builtin_qft3())
    print(qft_3())

    # test with both eigenvectors of U
    assert np.allclose(qpe('0'), builtin_qpe('0'))
    print(qpe('0'))
    assert np.allclose(qpe('1'), builtin_qpe('1'))
    print(qpe('1'))

    # should get the right eigenvalues
    assert results_to_eigenvalue(qpe('0')) == 1
    print('lambda_0:' + str(results_to_eigenvalue(qpe('0'))))
    assert results_to_eigenvalue(qpe('1')) == (-0.7071067811865477 - 0.7071067811865475j)
    print('lambda_1:' + str(results_to_eigenvalue(qpe('1'))))

    # convert to Clifford + T circuit
    print(qml.clifford_t_decomposition(qpe('0'), 10**-7))
    print(convert_to_clifford_T(qpe('0'), 10**-7))
