import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial


class GP:
    def __init__(self, n, mesh_size, num_funcs, lower_bound, upper_bound):
        '''
        :param n          :     The dimension of input data
        :param mesh_size  :     Specify the mesh size for each dimension
        :param num_funcs  :     The number of prior distribution functions
        :param lower_bound:     The lower bound of n-dimensional input, n by 1
        :param upper_bound:     The upper bound of n-dimensional input, n by 1
        '''
        self.n = n
        # xE stores the normalized evaluated data
        self.xE = np.empty(shape=[n, 0])
        # yE stores the values of utility function at evaluated data
        self.yE = np.empty(shape=[1, ])
        assert lower_bound.ndim == 2, "The input lower bound should be a 2D column-vector."
        assert upper_bound.ndim == 2, "The input upper bound should be a 2D column-vector."
        assert lower_bound.shape[0] == self.n, "The dimension of lower bound is incorrect."
        assert upper_bound.shape[0] == self.n, "The dimension of upper bound is incorrect."
        for i in range(self.n):
            assert lower_bound[i] > upper_bound[i], 'The %i-th value of lower bound is bigger than uppers ' % int(i)
        self.lb = lower_bound
        self.ub = upper_bound
        self.mesh_size = int(mesh_size)
        self.num_funcs = int(num_funcs)
        # self.X stores the mesh grid points, n by (mesh_size + 1)^n matrix
        self.X = self.meshgrid_generator()
        # The priori distribution
        self.mean = np.zeros(((mesh_size + 1) ** self.n, 1))
        self.cov = self.squared_exponential_kernel(self.X, self.X)

    def normalize_data(self, x):
        assert x.ndim == 2, 'The input x should be a 2D matrix, each column stores 1 data point.'
        data_range = self.ub - self.lb
        normalized_x = np.zeros(x.shape)
        for i in range(x.shape[1]):
            normalized_x[:, i] = (x[:, i].reshape(-1, 1) - self.lb) / data_range
        return normalized_x

    def physical_data(self, x):
        assert x.ndim == 2, 'The input x should be a 2D matrix, each column stores 1 data point.'
        data_range = self.ub - self.lb
        normalized_x = np.zeros(x.shape)
        for i in range(x.shape[1]):
            normalized_x[:, i] = x[:, i].reshape(-1, 1) * data_range + self.lb
        return normalized_x

    def update(self, x, y):
        '''
        Update (or, initialize) the sampled x and y into class.
        :param x:   Positions of sampled sites
        :param y:   The function values at sampled sites
        :return:    Update the class.
        '''
        assert x.ndim == 2, 'The input x should be a 2D matrix, each column stores 1 data point.'
        assert x.shape[0] == self.n, 'The input x should be n-dimensional data.'
        assert y.ndim == 1, 'The input y should be a 1D vector, each column stores 1 value for each data point.'
        if self.xE.shape[1] == 0:
            self.xE = np.copy(self.normalize_data(x))
            self.yE = np.copy(y)
        else:
            self.xE = np.hstack((self.xE, x))
            self.yE = np.hstack((self.yE, y))

    def squared_exponential_kernel(self, X, Y):
        '''
        K(x, x') = sigma_0^2 * exp[-1/2 * lambda * ||x-x'||^2]
        :param X: Could be matrix or vector, n by m1.
        :param Y: Could be matrix or vector, n by m2.
        :return : Covariance matrix of X and Y, the ij-th entry denote the covariance b/w
                  X[:, i] and Y[:, j].
        '''
        assert X.shape[0] == self.n, 'X should have the same size as n'
        assert Y.shape[0] == self.n, 'Y should have the same size as n'
        m1 = X.shape[1]
        m2 = Y.shape[1]
        cov = np.zeros((m1, m2))
        for i in range(m1):
            diff = (X[:, i]).reshape(-1, 1) - Y
            cov[i, :] = np.exp(-0.5 * np.diag(np.dot(diff, diff.T)))
        return cov

    def meshgrid_generator(self):
        '''
        -Generate mesh grid for multiple dimensions.
        - Notice that the order of rows in positions is reversed.
        - To recover the original dim, reverse search the positions matrix.
        :param grid_dim:  A dictionary that stores undefined number of inputs of mesh grid on each directions.
                          - Example -   x = np.linspace(0,1,9)
                                        y = np.linspace(0,1,5)
                                        z = np.linspace(0,1,17)
                                        Let grid['0'] = x, grid['1'] = y, grid['2'] = z.
                                        Run "g = meshgrid_generator(x, z, y)" gives the mesh grid matrix.
                                        "positions = np.vstack(map(np.ravel, g))" gives the mesh grid points in positions
        :param n       :  Dimension of inputs.
        :return:          The mesh grid in each dim stored in grid_dim columnwise. Shape as n by number of points.
        '''
        arrs = []
        for i in range(self.n):
            arrs.append(np.linspace(0, 1, self.mesh_size + 1))
        arrs = tuple(arrs)

        arrs = tuple(reversed(arrs))
        lens = list(map(len, arrs))
        dim = len(arrs)
        sz = 1
        for s in lens:
            sz *= s
        ans = []
        for i, arr in enumerate(arrs):
            slc = [1] * dim
            slc[i] = lens[i]
            arr2 = np.asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2)
        g = tuple(ans)
        # The order of rows is reversed
        positions = np.vstack(map(np.ravel, g))
        mesh_grid = np.zeros(positions.shape)
        for i in range(dim):
            mesh_grid[i, :] = np.copy(positions[dim - i - 1, :])

        return mesh_grid

    def get_prior(self):
        '''
        Generate the function values of prior distribution.
        Usually this function aims for results display. As a result,
        this function is only for 1D and 2D output.

        For higher-dimensions, i.e. self.n > 2, it is easier to directly compute the posterior distribution.
        :return:    The function values at mesh grid points.
        '''
        assert self.n > 2, "This prior is only for prior displaying, dimension n can not be more than 2."

        # Generate the mesh grid points, stored as n by Mesh_size^n matrix.
        X = self.meshgrid_generator()
        # Generate kernelized covariance matrix
        sigma = self.squared_exponential_kernel(X, X)

        # Generate random functions on mesh grid points:

        # Easier way using np.random.multivariate_normal:
        # Notice that the mean for prior is all zeros for now.
        # The shape o f Y would be (num_funcs) by (mesh_size + 1)^n matrix.
        Y = np.random.multivariate_normal(mean=np.zeros(X.shape[1]), cov=sigma, size=self.num_funcs)

        # # Self code way to generate Multivariate Gaussian distribution. Refer to Rasmussen & Williams Page-201.
        # U, S, V = np.linalg.svd(sigma)
        # # np.linalg.svd returns V.T as V
        # # Sigma = U S V.T, and J = V S^1/2 V.T
        # J = np.dot(V.T, np.dot(np.diag(np.sqrt(S)), V))
        # # Notice that the mean for prior is all zeros for now.
        # mean = np.zeros(X.shape[1]).reshape(-1, 1)
        # std = np.random.multivariate_normal(mean=np.zeros(X.shape[1]),
        #                                     cov=np.identity(X.shape[1]), size=self.num_funcs)
        # Y = (mean + np.dot(J, std.T)).T
        return mean, sigma

        if self.n == 2:
            prior_output = np.zeros((self.num_funcs, self.mesh_size + 1, self.mesh_size + 1))
            for i in range(self.num_funcs):
                prior_output[i, :, :] = Y[i, :].reshape(self.mesh_size + 1, self.mesh_size + 1)
        else:  # n == 1
            prior_output = np.copy(Y)

        return prior_output

    def visualizer_prior_1D(self):
        '''
        Generate the 1D plot of multivariate gaussian distribution, with the mean and covariance stored in self.mean
        and self.cov.
        The prior distribution info are generated at the beginning of class initialization.
        The posterior distribution are updated
        :return:
        '''
        Y = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.num_funcs)

        # # Self code way to generate Multivariate Gaussian distribution. Refer to Rasmussen & Williams Page-201.
        # U, S, V = np.linalg.svd(self.cov)
        # # np.linalg.svd returns V.T as V
        # # Sigma = U S V.T, and J = V S^1/2 V.T
        # J = np.dot(V.T, np.dot(np.diag(np.sqrt(S)), V))
        # # Notice that the mean for prior is all zeros for now.
        # std = np.random.multivariate_normal(mean=self.mean,
        #                                     cov=np.identity(self.mesh_size + 1), size=self.num_funcs)
        # Y = (self.mean + np.dot(J, std.T)).T

        plt.figure()
        for i in range(self.num_funcs):
            plt.scatter(self.X, Y[i, :])
            plt.plot(self.X, Y[i, :])
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('y = f(x)')
        if self.xE.shape[1] != 0:
            # There is evaluated data, make the posterior plot
            plt.scatter(self.xE.T[0], self.yE, marker='+', c='k')
        plt.show()
        return

    def visualizer_prior_2D(self, x, mean, cov):
        x = np.linspace(0, 1, self.mesh_size + 1)
        y = np.linspace(0, 1, self.mesh_size + 1)
        X, Y = np.meshgrid(x, y)
        prior_output = self.get_prior()
        for i in range(self.num_funcs):
            plt.figure()
            plt.contourf(X, Y, prior_output[i, :, :])
            plt.show()
        return

    def posterior(self, X_test):
        '''
        Calculate the mean and covariance for posterior distribution matrix at X_test positions, given the observations
        self.x and self.y.
        :param X_test:      The test input
        :return:            The mean and covariance for posterior distribution
        '''
        # TODO fix training output and input
        cov_11 = self.squared_exponential_kernel(self.x, self.x)
        cov_12 = self.squared_exponential_kernel(self.x, X_test)
        cov_22 = self.squared_exponential_kernel(X_test, X_test)
        # The transpose in the following line is a PUNCHLINE.
        # The covariance for conditional distribution is Sigma12 * Sigma11^(-1).
        cov12_cov11_inv = np.linalg.solve(cov_11, cov_12).T
        self.mean = np.dot(cov12_cov11_inv, self.y)
        self.cov = cov_22 - np.dot(cov12_cov11_inv, cov_12)

    def plot

if __name__ == "__main__":


