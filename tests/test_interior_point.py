import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
P=Path(__file__).resolve().parents[1]/'experiments/interior_point/benchmark_ip.py'
spec=importlib.util.spec_from_file_location('ip_target',P)
ip=importlib.util.module_from_spec(spec);spec.loader.exec_module(ip)

class TestInteriorPoint(unittest.TestCase):
    def test_barrier_derivatives(self):
        for family in ('generic','correlated','opposed'):
            ev,_,c=ip.problem(79,family);x=np.zeros(6)
            for mu in (.001,.1,1.):
                val,g,H,lam=ip.barrier(ev,x,c,mu)
                ng=approx_derivative(lambda y:ip.barrier(ev,y,c,mu)[0],x).ravel()
                nh=approx_derivative(lambda y:ip.barrier(ev,y,c,mu)[1],x)
                np.testing.assert_allclose(ng,g,atol=2e-5,rtol=1e-5)
                np.testing.assert_allclose(nh,H,atol=2e-5,rtol=1e-5)
                np.testing.assert_allclose(lam*(-ev(x)[3]),mu)
    def test_active_solution(self):
        def ev(x):
            return .5*(x[0]-2)**2,np.array([x[0]-2]),np.ones((1,1)),np.array([x[0]-1]),np.ones((1,1))
        r=ip.solve(ev,np.array([0.]),np.zeros(1),'baseline',.2,1)
        self.assertTrue(r['success']);self.assertAlmostEqual(r['x'][0],1.,places=4)
        self.assertAlmostEqual(r['dual'][0],1.,places=4)
    def test_reject_noninterior(self):
        ev,x,c=ip.problem(79,'generic')
        with self.assertRaises(ValueError):ip.barrier(ev,np.ones(6)*100,c,.1)
    def test_interior_optimum(self):
        def ev(x):
            return .5*(x@x),x.copy(),np.eye(2),np.array([-1.]),np.zeros((1,2))
        r=ip.solve(ev,np.array([.2,.3]),np.zeros(1),'baseline',.2,1)
        self.assertTrue(r['success']);self.assertLess(np.linalg.norm(r['x']),1e-4)

if __name__=='__main__':unittest.main()
