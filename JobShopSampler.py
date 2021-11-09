import numpy as np
import os

class JobShopSampler():
    def __init__(self, path, nJobs=None, nMachines=None, pmin=None, pmax=None):
        self._path = path
        self._J = nJobs
        self._m = nMachines
        self._pmin = pmin
        self._pmax = pmax
    def sample(self, nJobs=None, nMachines=None, pmin=None, pmax=None, nsamples = 1):
        path = self._path
        self._J = nnull_condition(self._J, nJobs)
        self._m = nnull_condition(self._m, nMachines)
        self._pmin = nnull_condition(self._pmin, pmin)
        self._pmax = nnull_condition(self._pmax, pmax)

        if nsamples > 1:
            dirname = "Samples"+str(nJobs)+"x"+str(nMachines)
            path += "/" + dirname + "/"
            if not os.path.exists(path):
                os.makedirs(path)
        num_equivalent_samples = 0
        for file in os.listdir(path):
            probsize = file[2:-4].split("_")
            sample_num = int(probsize[2])
            if probsize[0:1] == [str(self._J), str(self._m)] and sample_num > num_equivalent_samples:
                num_equivalent_samples = sample_num

        for nsample in range(nsamples):
            filename = path + "js" + str(self._J) + "x" + str(self._m) + "_" + str(num_equivalent_samples+nsample) \
                       + ".txt"
            M = self.sample_M()
            P = self.sample_P()
            file = open(filename, 'w+')
            file.write("".join([str(M[i, o])+" "+str(P[i,o]) for i, o in np.ndindex(self._J, self._m)]))
            file.close()

    def sample_M(self):
        return [np.random.permutation(np.arange(self._m)) for i in range(self._J)]

    def sample_P(self):
        return np.random.randint(self._pmin, self._pmax, size=(self._J, self._m))


def nnull_condition(a, b):
    if b is None:
        assert a is not None
        return a
    else:
        return b

