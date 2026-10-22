"""Numerical and import-safety tests for the exploratory ALM pilot."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative

P=Path(__file__).resolve().parents[1]/'experiments/alm/benchmark_alm.py'
spec=importlib.util.spec_from_file_location('alm_test_target',P)
alm=importlib.util.module_from_spec(spec);spec.loader.exec_module(alm)

class TestALM(unittest.TestCase):
    def test_augmented_gradient(self):
        for family in ('generic','correlated','opposed'):
            ev,x,_=alm.problem(79,family)
            for rho in (.3,10.,100.):
                lam=np.linspace(0,2,8)
                numerical=approx_derivative(lambda y:alm.augmented(ev,y,lam,rho)[0],x).ravel()
                np.testing.assert_allclose(numerical,alm.augmented(ev,x,lam,rho)[1],atol=2e-6,rtol=1e-6)
    def test_unconstrained_interior(self):
        def ev(x):
            return .5*(x@x),x.copy(),np.eye(2),np.array([-1.]),np.zeros((1,2))
        r=alm.solve(ev,np.array([2.,-1.]),'baseline',5.,1)
        self.assertTrue(r['success']);self.assertLess(np.linalg.norm(r['x']),1e-5)
    def test_active_inequality(self):
        # min .5(x-2)^2 subject to x<=1: x*=1, lambda*=1.
        def ev(x):
            return .5*(x[0]-2)**2,np.array([x[0]-2]),np.ones((1,1)),np.array([x[0]-1]),np.ones((1,1))
        r=alm.solve(ev,np.array([0.]),'baseline',5.,1)
        self.assertTrue(r['success']);self.assertAlmostEqual(r['x'][0],1.,places=5)
        self.assertAlmostEqual(r['dual'][0],1.,places=4)
    def test_complementarity_required(self):
        def ev(x):
            return 0.,np.array([-1.]),np.zeros((1,1)),np.array([-1.]),np.ones((1,1))
        c=alm.certificate(ev,np.zeros(1),np.ones(1))
        self.assertEqual(c['stationarity'],0.);self.assertFalse(c['success'])

if __name__=='__main__':unittest.main()
