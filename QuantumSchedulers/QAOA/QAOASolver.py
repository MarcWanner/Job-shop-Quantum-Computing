from QuantumSchedulers.QAOA.CircuitBuilders.QuboCircuitBuilder import QuboCircuitBuilder
from QuantumSchedulers.QAOA.QCSamplers.QiskitSimulator import QiskitSimulator
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import QiskitMinimizer, get_expectation, expected_value, \
    key_to_vector
from QuantumScheduler import QCJobShopScheduler, JobShopSchedulingData, HamiltonianConstructor
from QuantumSchedulers.QAOA.QAOA import Preprocessor, CircuitBuilder, QCSampler, ThetaOptimizer, Postprocessor
from random import random
import math
import matplotlib.pyplot as plt
import dimod
import numpy as np
from qiskit.visualization import plot_histogram


class QAOASolver(QCJobShopScheduler):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, p: int, theta=None, preprocessor: Preprocessor = None,
                 circuit_builder: CircuitBuilder = QuboCircuitBuilder(), qc_sampler: QCSampler = QiskitSimulator(),
                 theta_optimizer: ThetaOptimizer = QiskitMinimizer('COBYLA'), postprocessor: Postprocessor = None,
                 order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                 variable_pruning: bool = False):
        super().__init__(scheduling_data, hamiltonian_constructor, time_span, order_bias, machine_bias,
                                       single_run_bias, objective_bias, variable_pruning)
        self._preprocessor = preprocessor
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._theta_optimizer = theta_optimizer
        self._postprocessor = postprocessor
        self._p = p
        self._theta = theta
        if theta is None and preprocessor is None:
            self._theta = default_init_theta(p)
        self._qaoa_data = None
        self._quantum_circuit = None
        self._counts = None

    def solve(self, num_reads=100, energy_rank=0):
        if self._preprocessor is not None:
            self.process_qaoa_data(self._preprocessor.preprocess(self._hamiltonian, self._data))
        # adjust circuit builder so that it only depends on theta
        self._circuit_builder.set_bqm(self._hamiltonian, self._num_qubits)
        print(self._theta)
        # run QAOA
        self._theta = self._theta_optimizer.get_theta(self._hamiltonian, self._theta, num_reads, self._circuit_builder,                                                      self._qc_sampler)
        print(self._theta)
        qc = self._circuit_builder.get_quantum_circuit(self._theta, self._hamiltonian, self._num_qubits)
        self._counts = self._qc_sampler.sample_qc(qc, num_reads)
        #plot_histogram(self._counts)
        print(expected_value(self._counts, num_reads, self._hamiltonian))
        print(len(self._counts))
        if self._postprocessor is not None:
            postprocessing_input = None  # tbd, postprocessing input is a dummy, replace by arguments like qc, etc when
                                         # known what is needed
            self._counts = self._postprocessor.get_postprocessed_data(postprocessing_input)
        self._sampleset = to_sampleset(self._counts, self._hamiltonian)
        #print(self._sampleset)

    def process_qaoa_data(self, qaoa_data):
        pass

    def get_solver_name(self):
        return "QAOA"

    def plot_expectation_heatmap(self, shape, num_reads):
        self._circuit_builder.set_bqm(self._hamiltonian, self._num_qubits)
        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, num_reads)
        beta_stepsize = math.pi/shape[0]
        gamma_stepsize = 2*math.pi/shape[1]
        result = np.zeros(shape)
        for i, j in np.ndindex(shape):
            result[i, j] = expectation([i*beta_stepsize, j*gamma_stepsize])
            print(i * beta_stepsize, j * gamma_stepsize, result[i, j])
        plt.imshow(result)


def default_init_theta(p):
    return [math.pi * (1 + (i % 2)) * random() for i in range(2*p)]


def key_to_dict(key: str):
    res_dict = {}
    for i in range(len(key)):
        res_dict[i] = int(key[i])
    return res_dict


def to_energy_counts(counts: dict, hamiltonian):
    energy_counts = []
    for key, count in counts.items():
        x = key_to_vector(key)
        energy = np.dot(np.dot(x, hamiltonian), x)
        energy_counts.append((key_to_dict(key), energy, count))

    return energy_counts


def to_sampleset(counts: dict, hamiltonian):
    energy_counts = to_energy_counts(counts, hamiltonian)
    energy_counts.sort(key=lambda x: x[1])
    variables = [energy_count[0] for energy_count in energy_counts]
    energies = [energy_count[1] for energy_count in energy_counts]
    num_ocs = [energy_count[2] for energy_count in energy_counts]

    return dimod.SampleSet.from_samples(variables, 'BINARY', energies, num_occurrences=num_ocs)

