from abc import ABCMeta, abstractmethod
import sys
sys.path.append('../..')
import time
from qiskit import QuantumCircuit
import numpy as np
import pandas as pd
import seaborn as sns
import copy


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._hamiltonian = 0
        # unfortunately have to extend qaoa_data with None entries for each different preprocess approach
        # but this allows for preprocessing that affects different parts of the QAOA pipeline
        self._hamiltonian = None
        self._scheduling_data = None
        self._qaoa_data: dict = {"CONTINUOUS_SOLUTION": None}
        self._qaoa_data_name = None
        self._time = 0

    def get_preprocess_data(self, hamiltonian=None, scheduling_data=None):
        start = time.time()
        if self._qaoa_data[self._qaoa_data_name] is None:
            assert not (hamiltonian is None and self._hamiltonian is None)
            assert not (scheduling_data is None and self._scheduling_data is None)
            self.preprocess(hamiltonian, scheduling_data)
        end = time.time()
        self._time = end - start
        return self._qaoa_data

    @abstractmethod
    def preprocess(self, hamiltonian=None, scheduling_data=None):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_time(self):
        return self._time


class CircuitBuilder(metaclass=ABCMeta):
    def __init__(self, log_qc=False):
        self._quantum_circuit = None
        self._theta = None
        self._bqm = None
        self._num_qubits = None
        self._qaoa_data: dict = None
        self._time = 0
        self._nreps = 0
        self._log_qc = log_qc
        self._mixer = None
        self._problem = None

    def get_quantum_circuit(self, theta=None, bqm=None, num_qubits=None, qaoa_data=None):
        start = time.time()
        if qaoa_data is not None:
            self.set_preprocess_data(qaoa_data)
        if self._quantum_circuit is None:
            assert theta is not None
            assert not(self._bqm is None and (bqm is None or num_qubits is None))
            if bqm is None:
                bqm = self._bqm
                num_qubits = self._num_qubits
            self.build_quantum_circuit(theta, bqm, num_qubits)
        elif theta is not None:
            if bqm is None:
                bqm = self._bqm
                num_qubits = self._num_qubits
            else:
                assert num_qubits is not None
            self.build_quantum_circuit(theta, bqm, num_qubits)
        end = time.time()
        self._time += end - start
        self._nreps += 1
        return self._quantum_circuit

    def set_bqm(self, bqm, num_qubits):
        self._bqm = bqm
        self._num_qubits = num_qubits
        
    def get_bqm(self):
        return self._bqm

    def set_preprocess_data(self, qaoa_data: dict):
        self._qaoa_data = qaoa_data

    def get_time(self):
        return self._time/self._nreps

    def reset_time(self):
        self._time = 0
        self._nreps = 0

    def get_mixer(self):
        return self._mixer

    def get_problem(self):
        return self._problem

    def plot_annealing(self, evals=[0], tsteps=1000):
        assert self._log_qc == True
        H_init = self.get_mixer()
        dH = (self.get_problem() - H_init)
        evalmat = np.zeros((len(evals) + 1, tsteps))
        dt = 1/tsteps
        columns = ["Energy", "t", "eigenvalue"]
        df_data = []
        for t in range(tsteps):
            H_t = H_init + t * dH * dt
            all_eigvals, all_eigvecs = np.linalg.eigh(H_t)
            eigvals = all_eigvals[evals]
            eigvecs = all_eigvecs[evals, :]
            adiabatic_thm_values = np.zeros(len(eigvecs) - 1)
            grad_e0 = None
            #change this -- very inefficient and numerical instabilities
            """ 
            if t == 0:
                H_t_plus1 = H_init + (t + 1) * dH * dt
                eigvals2, eigvecs2 = np.linalg.eigh(H_t_plus1)
                grad_e0 = (eigvecs2[0, :] - eigvecs[0, :]) / dt
            elif t == tsteps - 1:
                H_t_minus1 = H_init + (t - 1) * dH * dt
                eigvals1, eigvecs1 = np.linalg.eigh(H_t_minus1)
                grad_e0 = (eigvecs[0, :] - eigvecs1[0, :]) / dt
            else:
                H_t_plus1 = H_init + (t + 1) * dH * dt
                eigvals2, eigvecs2 = np.linalg.eigh(H_t_plus1)
                H_t_minus1 = H_init + (t - 1) * dH * dt
                eigvals1, eigvecs1 = np.linalg.eigh(H_t_minus1)
                grad_e0 = (eigvecs2[0, :] - eigvecs1[0, :]) / (2 * dt)
            for i in range(1):
                adiabatic_thm_values[i] = np.abs(np.dot(eigvecs[i + 1], grad_e0) / (eigvals[0] - eigvals[i + 1]))
            adiabatic_thm_value = np.max(adiabatic_thm_values)
            """
            #print("All_evals: ", np.linalg.eigvalsh(H_t), " only subset of evals: ", eigvals)
            for k in range(len(eigvals)):
                df_data.append([np.real(eigvals[k]), t * dt, f"$E_{evals[k]}$"])
            #numerical instabilities
            #df_data.append([adiabatic_thm_value, t * dt, "adiabatic thm value"])

        df = pd.DataFrame(df_data, columns=columns)
        df.astype({"Energy": "float64", "t": "float64", "eigenvalue": "str"})
        sns.set_style('darkgrid')
        sns.lineplot(data=df, x='t', y="Energy", hue="eigenvalue").set_title("Energy spectrum of adiabatic evolution")

    @abstractmethod
    def build_quantum_circuit(self, theta, bqm, num_qubits: int):
        pass

    @abstractmethod
    def get_name(self):
        pass


class QCSampler(metaclass=ABCMeta):
    def __init__(self, seed_simulator=937162211):
        self._backend = None
        self._seed = seed_simulator
        self._time = 0
        self._nreps = 0

    def get_counts(self, quantum_circuit, num_reads):
        start = time.time()
        counts = self.sample_qc(quantum_circuit, num_reads)
        end = time.time()
        self._time += end - start
        self._nreps += 1
        return counts

    @abstractmethod
    def sample_qc(self, quantum_circuit, num_reads):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_seed(self):
        return self._seed

    def get_time(self):
        return self._time/self._nreps

    def reset_time(self):
        self._time = 0
        self._nreps = 0

    def get_nreps(self):
        return self._nreps


class ThetaOptimizer(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._qc = 0
        self._theta = None
        self._circuit_builder = None
        self._qc_sampler = None
        self._num_reads = None
        self._hamiltonian = None
        self._expected_energy = None
        self._time = 0

    def get_theta(self, hamiltonian, theta_init, num_reads: int, circuit_builder=None, qc_sampler=None):
        start = time.time()
        if self._theta is None:
            assert not (circuit_builder is None and self._circuit_builder is None)
            assert not (qc_sampler is None and self._qc_sampler is None)
            self.optimize_theta(circuit_builder, qc_sampler, num_reads, hamiltonian, theta_init)
        elif theta_init is not None:
            cb = circuit_builder
            qs = qc_sampler
            if cb is None:
                cb = self._circuit_builder
            if qs is None:
                qs = self._qc_sampler
            self.optimize_theta(cb, qs, num_reads, hamiltonian, theta_init)
        end = time.time()
        self._time = end - start
        return self._theta

    def get_expected_energy(self):
        return self._expected_energy

    @abstractmethod
    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta_init):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_time(self):
        return self._time


class Postprocessor(metaclass=ABCMeta):
    def __init__(self):
        self._postprocessed_counts: dict = None
        self._postprocessing_input: dict = None
        self._time = 0

    def get_postprocessed_data(self, postprocessing_input: dict = None):
        start = time.time()
        if self._postprocessed_counts is None:
            assert postprocessing_input is not None
            self.postprocess(postprocessing_input)
        end = time.time()
        self._time = end - start
        return self._postprocessed_counts

    @abstractmethod
    def postprocess(self, postprocessing_input: dict):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_time(self):
        return self._time


class LogQuantumCircuit:
    def __init__(self, nqubits):
        self._matrix: np.ndarray = np.zeros((2**nqubits, 2**nqubits))
        self._nqubits = nqubits

    def rx(self, theta, index):
        self._matrix += 0.5 * theta * self.single_kron(np.array([[0, 1], [1, 0]]), index)

    def rz(self, theta, index):
        self._matrix += 0.5 * theta * self.single_kron(np.array([[1, 0], [0, -1]]), index)

    def rzz(self, theta, index1, index2):
        #very bad and ineficient
        sigma1 = self.single_kron(np.array([[1, 0], [0, -1]]), index1)
        sigma2 = self.single_kron(np.array([[1, 0], [0, -1]]), index2)
        self._matrix += 0.5 * theta * np.matmul(sigma1, sigma2)

    def ry(self, theta, index):
        self._matrix += 0.5 * theta * self.single_kron(np.array([[0, -1j], [1j, 0]]), index)

    def single_kron(self, single_op, index):
        res = np.identity(2)
        if index == 0:
            res = single_op
        for i in range(1, self._nqubits):
            op = np.identity(2)
            if i == index:
                op = single_op

            res = np.kron(res, op)
        return res

    def get_hamiltonian_matrix(self):
        return self._matrix

    def append(self, qc):
        self._matrix += qc.get_hamiltonian_matrix()









