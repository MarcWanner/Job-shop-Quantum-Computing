from Reader import JobShopReader
from HamiltonianConstructor import JobShopHamiltonianConstructor
from QuantumScheduler import SASolver, QASolver, QAOASolver
from Scheduler import CPLEXSolver
import matplotlib.pyplot as plt
from qiskit import Aer, transpile
from QuantumSchedulers.QAOA.QAOASolver import expected_value
import numpy as np

def main():

    reader = JobShopReader()
    reader.read_problem_data("Problems/micro_example.txt")
    data = reader.get_data()

    solvers = []
    #solvers = [CPLEXSolver(data)]
    #solvers.append(SASolver(data, JobShopHamiltonianConstructor(), 5, variable_pruning=True))
    #solvers.append(QASolver(data, JobShopHamiltonianConstructor(), 8, variable_pruning=True))
    solvers.append(QAOASolver(data, JobShopHamiltonianConstructor(), 5, 1, variable_pruning=True, objective_bias=0,
               theta=[1, 1]))
    for solver in solvers:
        #solver.solve(num_reads=1000)
        #solver.plot_solution()
        solver.plot_expectation_heatmap((50, 50), 512)
    plt.show()


    """
    qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian, qaoa_solver._num_qubits)
    #qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
                                         #qaoa_solver._num_qubits)
    thetas = [[0.5, 1], [1, 1], [2, 2], [2, 1], [2, 5]]
    for theta in thetas:

        qc = qaoa_solver._circuit_builder.get_quantum_circuit(theta=theta)
        qc.decompose().decompose().draw(output='mpl')
        plt.show()
        print(qaoa_solver._qc_sampler.sample_qc(qc, 512))
        backend = Aer.get_backend('aer_simulator')
        qobj = transpile(qc, backend)
        counts = backend.run(qobj, validate=True, seed_simulator=7, shots=512).result().get_counts(qc)
        print(expected_value(counts, 512, qaoa_solver._hamiltonian))
        #qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
                                             #qaoa_solver._num_qubits)
    """
    #qaoa_solver.plot_expectation_heatmap((25, 50), 1000)
    #qaoa_solver.solve()
    #qaoa_solver.plot_solution()
    #plt.show()


if __name__ == "__main__":
    main()