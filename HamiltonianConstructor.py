from abc import ABCMeta, abstractmethod
from data import SchedulingData, JobShopSchedulingData

class HamiltonianConstructor(metaclass=ABCMeta):
    def __init__(self):
        self._data = None
        self._hamiltonian = None

    def get_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              variable_pruning: bool = False):
        if self._data is None:
            assert scheduling_data is not None and time_span is not None
            self.construct_hamiltonian(scheduling_data, time_span, order_bias, machine_bias, single_run_bias,
                                       objective_bias, variable_pruning)
        return self._hamiltonian

    @abstractmethod
    def construct_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              qubit_impugning: bool = False):
        pass

    @abstractmethod
    def get_plottable_solution(self, solution_sampleset, energy_rank=0):
        pass

    @abstractmethod
    def plottable_solution_to_pruned(self, plottable_solution):
        pass

from HamiltonianConstructors.JobShopHamiltonianConstructor import JobShopHamiltonianConstructor