from QuantumSchedulers.QAOA.QAOA import CircuitBuilder, LogQuantumCircuit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from dimod import BinaryQuadraticModel
import matplotlib.pyplot as plt

class QuboCircuitBuilder(CircuitBuilder):
    def __init__(self, log_qc=False):
        super().__init__(log_qc)

    def build_quantum_circuit(self, theta, bqm, num_qubits: int):
        self._theta = theta
        self._bqm = bqm
        self._num_qubits = num_qubits
        p = int(len(theta) / 2)
        self._quantum_circuit = QuantumCircuit(num_qubits)
        if self._log_qc and self._mixer is None and self._problem is None:
            self._mixer = u_mixer(0, self._num_qubits, True).get_hamiltonian_matrix()
            self._problem = u_problem_dense(0, self._num_qubits, self._bqm, True).get_hamiltonian_matrix()

        betas = theta[:p]
        gammas = theta[p:]
        u_problem = u_problem_dense

        if type(bqm).__name__ is 'BinaryQuadraticModel':
            u_problem = u_problem_sparse

        self._quantum_circuit.append(qc_init(num_qubits), range(num_qubits))

        for i in range(p):
            self._quantum_circuit.append(u_problem(gammas[i], num_qubits, bqm), range(num_qubits))
            self._quantum_circuit.append(u_mixer(betas[i], num_qubits), range(num_qubits))

        self._quantum_circuit.measure_all()

    def get_name(self):
        return "QUBOCIRCUITBUILDER"


def qc_init(num_qubits: int):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc


def u_mixer(beta: float, num_qubits: int, log_qc=False):
    qc = QuantumCircuit(num_qubits)
    if log_qc:
        qc = LogQuantumCircuit(num_qubits)
        beta = 1
    for i in range(num_qubits):
        qc.rx(-2*beta, i)
    return qc


def u_problem_dense(gamma: float, num_qubits: int, hamiltonian, log_qc=False):
    qaoa_circuit = QuantumCircuit(num_qubits)
    if log_qc:
        qaoa_circuit = LogQuantumCircuit(num_qubits)
        gamma = 1
    # Apply R_Z rotational gates from cost layer
    for j in range(num_qubits):
        sumq = 0
        for k in range(num_qubits):
            sumq += 0.5 * (hamiltonian[j, k] + hamiltonian[k, j])
        if sumq != 0:
            qaoa_circuit.rz(-sumq * gamma, j)

    # Apply R_ZZ rotational gates for entangled qubit rotations from cost layer
    for j in range(num_qubits):
        for k in range(num_qubits):
            if k == j or hamiltonian[j, k] == 0:
                continue
            else:
                qaoa_circuit.rzz(0.5 * hamiltonian[j, k] * gamma, j, k)

    return qaoa_circuit


# iterate over the bqm, i.e. the not 0 indices
def u_problem_sparse(gamma: float, num_qubits: int, bqm):
    pass


