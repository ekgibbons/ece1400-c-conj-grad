import sys
import time

import numpy as np
from scipy import io as spio


def solve(A: np.array, b: np.array) -> np.array:
    """Computes the conjugate gradient algorithm as described 
    in Algorithm in the homework document

    Parameters
    ----------
    A : np.array
        Input array A with size MxN (not square!)
    b : np.array
        Input vector b with size Mx1

    Returns
    -------
    np.array
        The solution array x with size Nx1

    Examples
    --------
    >>> A = np.random.uniform(-10,10,size=(5,3))
    >>> b = np.random.uniform(-10,10,size=(5,1))
    >>> x = solve(A,b)
    """

    # define the maximum number of iterations
    MAX_ITER = 10000
        
    # do some quick resizing to ensure the right
    # dimensions for b
    b = b.reshape(-1,1)

    # compute A^{\dagger} and b^{\dagger}
    ATA = A.T@A
    ATb = A.T@b

    # initialize x to all zeros as a starting guess
    x = np.zeros((A.shape[1],1))

    # line 1 (of algorithm 1)
    k = 0
    
    # line 2 (of algorithm 1)
    r = ATb - (ATA @ x)

    # compute the norm of the residual vector
    rTr_old = r.T@r
    
    # line 3 (of algorithm 1)
    if rTr_old < 1e-15:
        return x
    
    # line 6 (of algorithm 1)
    p = np.copy(r)
    
    # line 7 (of algorithm 1)
    while k < MAX_ITER:

        # precompute this term because you will use
        # it a few times
        Ap = ATA @ p

        # line 8 (of alorithm 1)
        alpha = rTr_old /((p.T) @ Ap)

        # line 9 (of algorithm 1)
        x += alpha * p
        # line 10 (of algorithm 1)
        r -= alpha * Ap
        
        # comute the norm of the updated residuals
        rTr_new = r.T @ r

        # line 11 (of algorithm 1)
        if rTr_new < 1e-15:
            break

        # line 14 (of algorithm 1)
        beta = rTr_new/rTr_old

        # line 15 (of algorithm 1)
        p = r + beta * p

        # reset the norm of residuals for the next
        # iteration through the loop
        rTr_old = np.copy(rTr_new)

        k += 1

    print(k)
        
    # return the solution
    return x


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage:\n    "
              "$ python %s <A_file> <b_file> <x_file>")
        sys.exit()

    # read in the data
    A = spio.mmread(sys.argv[1])
    b = spio.mmread(sys.argv[2])

    # run and time the solver
    print("solving %ix%i system" % A.shape)
    start = time.time()
    x = solve(A,b)
    stop = time.time()
    print("done...took %f seconds" % (stop - start))

    x_np = np.linalg.solve((A.T).dot(A), (A.T).dot(b))

    # write the data
    spio.mmwrite(sys.argv[3],x)


