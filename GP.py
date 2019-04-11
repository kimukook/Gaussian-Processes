import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial


class GP:
    def __init__(self, mesh_size, n, num_funcs, lower_bound, upper_bound):
        self.n = n
        self.x = np.empty(shape=[n, 0])
        self.y = np.empty(shape=[1,])
        self.lb = lower_bound
        self.ub = upper_bound
        self.mesh_size = mesh_size
        self.num_funcs = num_funcs

    def update(self, X):

        return

    def squared_exponential_kernel(self, X, Y):
        '''
        K(x, x') = sigma_0^2 * exp[-1/2 * lambda * ||x-x'||^2]
        :param X: Could be matrix or vector, n by m1.
        :param Y: Could be matrix or vector, n by m2.
        :return:
        '''
        # TODO: modify this to be vector calculation!
        assert X.shape[0] == self.n, 'X should have the same size as n'
        assert Y.shape[0] == self.n, 'Y should have the same size as n'
        m1 = X.shape[1]
        m2 = Y.shape[1]
        cov = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                diff = (X[:, i] - Y[:, j]).reshape(-1, 1)
                cov[i, j] = np.exp(-0.5 * np.dot(diff.T, diff))
        return cov

    def get_prior(self):
        X = np.linspace()
        return

    def normailize_data(self):

        return

    def physical_data(self):

        return

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
        :return:          The mesh grid matrix with the mesh grid generated in each dim, stored in grid_dim.
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



def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

nb_of_samples = 41  # Number of points in each function
number_of_functions = 5  # Number of functions to sample
# Independent variable samples
X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
sigma = exponentiated_quadratic(X, X)  # Kernel of data points

# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=sigma,
    size=number_of_functions)

for i in range(5):
    plt.plot(X.T[0], ys[i])
plt.show()

# 1D example
num = 101
X = np.expand_dims(np.linspace(-4, 4, num), 1)
sigma = exponentiated_quadratic(X, X)  # Kernel of data points
U, S, V = np.linalg.svd(sigma)
V = V.T
J = np.dot(V, np.dot(np.diag(np.sqrt(S)), V.T))
mu=np.random.multivariate_normal(mean=np.zeros(num), cov=np.identity(num))
mut = np.dot(J, mu)
y1=np.random.multivariate_normal(
    mean=np.zeros(num), cov=sigma)
plt.figure()
plt.plot(X, y1, label='good')
plt.plot(X, mut, label='self')
plt.legend()
plt.show()


def meshgrid_generator(n, mesh_size):
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
    :return:          The mesh grid matrix with the mesh grid generated in each dim, stored in grid_dim.
    '''
    arrs = []
    for i in range(n):
        arrs.append(np.linspace(0, 1, mesh_size + 1))
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


# 2D example
num = 50  # number of points for each dimension
X = meshgrid_generator(2, num)
sigma = exponentiated_quadratic(X.T, X.T)
U, S, V = np.linalg.svd(sigma)
V = V.T
J = np.dot(V, np.dot(np.diag(np.sqrt(S)), V.T))
mu = np.random.multivariate_normal(mean=np.zeros(X.shape[1]), cov=np.identity(X.shape[1]))
mean = np.zeros((X.shape[1], 1))
mut = mean + np.dot(J, mu.reshape(-1, 1))

y1 = np.random.multivariate_normal(
    mean=np.zeros(X.shape[1]), cov=sigma)

# plot y1 and mut, reshape 2601 to be 51 * 51.

