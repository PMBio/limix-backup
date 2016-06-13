import scipy.linalg as la
import numpy as np

class psd_solver(object):
    """a linear equation solver for symmetric positive semi-definite matrices"""

    def __init__(self, A, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False):
        self._s=None
        self._U=None
        self._chol=None
        self._lower=lower
        try:
            self._chol=la.cholesky(A,overwrite_a=overwrite_a,check_finite=True).T
    
        except la.LinAlgError:
            s,U = la.eigh(A,lower=lower)
            i_pos = (s>threshold)
            if i_pos.any():
                self._s = s[i_pos]
                self._U = U[:,i_pos]
            

            

            
    def solve(self,b,overwrite_b=False,check_finite=True):
        """
        solve A \ b
        """
        if self._s is not None:
            res = self._U.T.dot(b)
            res /= self._s[:,np.newaxis]
            res = self._U.dot(res)
        elif self._chol is not None:
            res = la.cho_solve((self._chol,self._lower),b=b,overwrite_b=overwrite_b,check_finite=check_finite)
        else:
            res = np.zeros(b.shape)
        return res

    def logdet(self):
        raise NotImplementedError("logdet not implemened yet.")

class psd_solver_any(object):
    """a linear equation solver for symmetric positive semi-definite matrices for the case where only any effects are present"""

    def __init__(self, A, lower=True, threshold=1e-10,check_finite=True,overwrite_a=False):
        self.solver = []
        (self.P,self.dof_any) = A.shape[0:2]
        self._lower=lower
        #This is trivially parallelizable:
        for p in range(A.shape[0]):
            self.solver.append(psd_solver(A[p]))

    def solve(self,b,overwrite_b=False,check_finite=True, p=None):
        """
        solve A \ b
        """
        if p is None:
            assert b.shape[:2]==(len(self.solver),self.dof_any)
            solution = np.empty(b.shape)
            #This is trivially parallelizable:
            for p in range(self.P):
                solution[p] = self.solver[p].solve(b=b[p])
            return solution
        else:
            return self.solver[p].solve(b=b)
    def logdet(self):
        raise NotImplementedError("logdet not implemened yet.")

class PsdSolverKron(object):
    """
    efficient general Kronecker solver for the case where any effects are present
    """

    def __init__(self, A_any, A, AA_any, lower=True, threshold=1e-10, check_finite=True, overwrite_a=False):
        self._lower=lower
        self.schur_solver = None
        self.DinvC = None
        self.A_any_solver = None

        if (A_any is not None) and A_any.shape[0]:#any effects are present
            self.A_any_solver = psd_solver_any(A=A_any, lower=lower, threshold=threshold,check_finite=check_finite,overwrite_a=overwrite_a)
            if (A is not None) and A.shape[0]>0:
                self.DinvC = self.A_any_solver.solve(AA_any)
                if overwrite_a:
                    schur = A
                    schur -= np.tensordot(AA_any,self.DinvC,axes=([0,1],[0,1]))
                else:
                    schur = A - np.tensordot(AA_any,self.DinvC,axes=([0,1],[0,1]))
                self.schur_solver = psd_solver(A=schur, lower=lower, threshold=threshold,check_finite=check_finite,overwrite_a=overwrite_a)
            
        elif (A is not None) and A.shape[0]>0:
            self.schur_solver = psd_solver(A=A, lower=lower, threshold=threshold,check_finite=check_finite,overwrite_a=overwrite_a)
          

    def solve(self, b_any, b, check_finite=True, p=None):
        """
        solve A \ b
        """
        #assert b.shape[:2]==(len(self.solver),self.dof_any)
        

        if self.schur_solver is None and self.A_any_solver is None:
            assert ( (b is None) or (b.shape[0]==0) ) and ( (b_any is None) or (b_any.shape[0]==0) ), "shape missmatch"
            return b, b_any
        elif self.schur_solver is None:
            assert (b is None) or (b.shape[0]==0), "shape missmatch"
            solution_any = self.A_any_solver.solve(b=b_any,p=p)
            return b,solution_any
        elif self.A_any_solver is None:
            assert (b_any is None) or (b_any.shape[0]==0), "shape missmatch"
            solution = self.schur_solver.solve(b=b, check_finite=check_finite)
            return solution, b_any
        else:
            assert p is None, "p is not None"
            cross_term = np.tensordot(self.DinvC,b_any,axes=([0,1],[0,1]))
            solution = self.schur_solver.solve(b=(b - cross_term), check_finite=check_finite)
            solution_any = self.A_any_solver.solve(b=b_any, check_finite=check_finite, p=p)
            solution_any -= self.DinvC.dot(solution)
            return solution, solution_any
