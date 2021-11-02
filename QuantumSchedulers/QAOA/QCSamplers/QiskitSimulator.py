from QuantumSchedulers.QAOA.QAOA import QCSampler
from qiskit import Aer, transpile


class QiskitSimulator(QCSampler):
    def __init__(self):
        super().__init__()

    def sample_qc(self, quantum_circuit, num_reads, simulator_type: str = None):
        if simulator_type is not None:
            self._backend = Aer.get_backend(simulator_type)
        elif self._backend is None:
            self._backend = Aer.get_backend('qasm_simulator')

        qobj = transpile(quantum_circuit, self._backend)
        counts = self._backend.run(qobj, seed_simulator=7, shots=num_reads).result().get_counts()

        return counts
