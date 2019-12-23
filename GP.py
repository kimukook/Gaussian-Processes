import numpy as np
import matplotlib.pyplot as plt
# A perfect guide could be found at: https://peterroelants.github.io/posts/gaussian-process-tutorial/
# Another perfect guidance: http://krasserm.github.io/2018/03/19/gaussian-processes/


class GaussianProcessRegression:
    def __init__(self, n, func, kernel_name, mesh_size, lower_bound, upper_bound, x=None, y=None):
        '''
        :param n          :     The dimension of input data
        :param func       :     The objective function
        :param kernal_name:     The name of the kernel function used to generate covariance matrix
        :param mesh_size  :     Specify the mesh size for each dimension
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
            assert lower_bound[i] < upper_bound[i], 'The %i-th value of lower bound is bigger than uppers ' % int(i)
        # Define the kernel function
        if kernel_name == 'SE':
            self.kernel_func = self.squared_exponential_kernel
        self.lb = lower_bound
        self.ub = upper_bound
        self.mesh_refine = 4
        self.mesh_size = int(mesh_size)
        self.max_iter = self.n * 20
        self.beta = 5

        self.func_eval = func
        # self.X stores the mesh grid points, n by (mesh_size + 1)^n matrix
        self.X = self.meshgrid_generator()
        # The priori distribution
        self.mean = np.zeros((mesh_size + 1) ** self.n)
        self.cov = self.squared_exponential_kernel(self.X, self.X)

        # Initialize the evalauted data points.
        # If it is not given, randomly select points within the bounds and evaluate those.
        if x is None:
            self.xE = self.lb + (self.ub - self.lb) * np.random.rand(self.n, 2)
        else:
            pass

        if y is None:
            self.yE = np.zeros(self.xE.shape[1])
            for i in range(self.xE.shape[1]):
                self.yE[i] = self.func_eval(self.xE[:, i])
        else:
            pass

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
        self.xE = np.hstack((self.xE, x))
        self.yE = np.hstack((self.yE, y))

    def squared_exponential_kernel(self, X, Y, l=1.0, sigma_f=1.0):
        '''
        K(x, x') = sigma_0^2 * exp[-1/2 * lambda * ||x-x'||^2]
        :param X: Could be matrix or vector, n by m1.
        :param Y: Could be matrix or vector, n by m2.
        :return : Covariance matrix of X and Y, the ij-th entry denote the covariance b/w
                  X[:, i] and Y[:, j].
        '''
        assert X.shape[0] == self.n, 'X should have the same size as n'
        assert Y.shape[0] == self.n, 'Y should have the same size as n'
        sq_dist = np.sum(X**2, 0).reshape(-1, 1) + np.sum(Y**2, 0) - 2 * np.dot(X.T, Y)
        return sigma_f**2 * np.exp(-1/(2*l**2) * sq_dist)

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
            arrs.append(np.linspace(self.lb[i], self.ub[i], self.mesh_size + 1))
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

    def posterior(self):
        '''
        Calculate the mean and covariance for posterior distribution matrix at X_test positions, given the observations
        self.x and self.y.
        :param X_test:      The test input, usually it is the mesh grid points of given n-dimensional parameter space.
        :return:            The mean and covariance for posterior distribution
        '''
        cov_11 = self.kernel_func(self.xE, self.xE)
        cov_12 = self.kernel_func(self.xE, self.X)
        cov_22 = self.kernel_func(self.X, self.X)
        # The transpose in the following line is a PUNCHLINE.
        # The covariance for conditional distribution is Sigma12 * Sigma11^(-1).
        cov12_cov11_inv = np.linalg.solve(cov_11, cov_12).T
        self.mean = np.dot(cov12_cov11_inv, self.yE)
        self.cov = cov_22 - np.dot(cov12_cov11_inv, cov_12)

    def optimizer(self):
        for kk in range(self.mesh_refine):
            for k in range(self.max_iter):
                self.posterior()
                predictive_distribution = self.mean + self.beta * np.diag(self.cov)
                x_iter_min = self.X[:, np.argmax(predictive_distribution)].reshape(-1, 1)
                y_iter_min = self.func_eval(x_iter_min)

                if np.min(np.linalg.norm(self.xE - x_iter_min, axis=0)) < 1e-6:
                    print(f'The point {x_iter_min.T[0]} has already been found')
                    if kk < self.mesh_refine - 1:
                        print('Refine the mesh...\n')
                        self.beta *= 2
                        self.mesh_size *= 2
                        self.X = self.meshgrid_generator()
                    break
                else:
                    self.xE = np.hstack((self.xE, x_iter_min))
                    self.yE = np.hstack((self.yE, y_iter_min))

    def visualizer_1D(self):
        '''
        Generate the 1D plot of multivariate gaussian distribution, with the mean and covariance stored in self.mean
        and self.cov.
        The prior distribution info are generated at the beginning of class initialization.
        The posterior distribution are updated
        :return:
        '''
        Y = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=8)

        # # Self code way to generate Multivariate Gaussian distribution. Refer to Rasmussen & Williams Page-201.
        # U, S, V = np.linalg.svd(self.cov)
        # # np.linalg.svd returns V.T as V
        # # Sigma = U S V.T, and J = V S^1/2 V.T
        # J = np.dot(V.T, np.dot(np.diag(np.sqrt(S)), V))
        # # Notice that the mean for prior is all zeros for now.
        # std = np.random.multivariate_normal(mean=self.mean,
        #                                     cov=np.identity(self.mesh_size + 1), size=self.num_funcs)
        # Y = (self.mean + np.dot(J, std.T)).T

        fig = plt.figure(figsize=[16, 9])
        for i in range(8):
            plt.scatter(self.X, Y[i, :].reshape(-1, 1), marker='o', s=3, zorder=i)
            plt.plot(self.X[0], Y[i, :])
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('y = f(x)')
        if self.xE.shape[1] > 0:
            # There is evaluated data, make the posterior plot
            plt.scatter(self.xE, self.yE.reshape(-1, 1), marker='+', c='k', s=50, zorder=8 + 1)
        plt.show()

    def confidence_bound_1D(self):
        fig = plt.figure(figsize=[16, 9])
        # plot the objective
        obj_x = np.arange(self.lb[0], self.ub[0]+0.02, 0.01).reshape(-1, 1).T
        obj_y = np.zeros(obj_x.shape[1])
        for i in range(obj_x.shape[1]):
            obj_y[i] = self.func_eval(obj_x[:, i])
        plt.plot(obj_x[0], obj_y, c='k', label=r'$f(x)$', zorder=0)
        # plot the surrogate
        plt.plot(self.X[0], self.mean, c='b', label=r"$\mu$(x)", zorder=0)
        # plot the evaluated points
        plt.scatter(self.xE, self.yE.reshape(-1, 1), marker='+', c='r', s=100, zorder=5, label=r'Evaluated points')
        # plot the confidence bound
        ucb = self.mean + 2 * np.diag(self.cov)
        lcb = self.mean - 2 * np.diag(self.cov)
        plt.fill_between(self.X[0], ucb, lcb, color='grey', alpha=0.5, label='Confidence Bound')
        plt.grid()
        plt.legend()
        plt.show()
        plt.close(fig)

    def visualizer_2D(self):
        x = np.linspace(self.lb[0, 0], self.ub[0, 0], self.mesh_size + 1)
        y = np.linspace(self.lb[1, 0], self.ub[1, 0], self.mesh_size + 1)
        X, Y = np.meshgrid(x, y)
        MultiNormal = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=1)
        Z = MultiNormal.reshape(self.mesh_size + 1, self.mesh_size + 1)

        plt.figure(figsize=[16, 9])
        plt.contourf(X, Y, Z, cmap='gray')
        plt.scatter(self.xE[0, :], self.xE[1, :], c='w', marker='s', s=70, edgecolors='k')
        plt.grid()
        plt.show()

        plt.figure(figsize=[16, 9])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.vstack((X[i, j], Y[i, j]))
                Z[i, j] = self.func_eval(point)
        plt.contourf(X, Y, Z, cmap='gray')
        plt.scatter(self.xE[0, :], self.xE[1, :], c='w', marker='s', s=70, edgecolors='k')
        plt.grid()
        plt.show()
