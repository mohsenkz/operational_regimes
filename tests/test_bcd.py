import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[1]/'experiments/bcd/benchmark_bcd.py'
spec=importlib.util.spec_from_file_location('bcd_pilot',P)
bcd=importlib.util.module_from_spec(spec);spec.loader.exec_module(bcd)

class TestBCD(unittest.TestCase):
    def setUp(self):
        self.obj,self.con,self.L,self.x=bcd.problem(412,bcd.CONDITIONS[0])
    def test_derivatives(self):
        rng=np.random.default_rng(19);x=rng.normal(size=len(self.x))*.15;h=1e-6
        _,g=self.obj(x)
        fd=np.array([(self.obj(x+np.eye(len(x))[j]*h)[0]-self.obj(x-np.eye(len(x))[j]*h)[0])/(2*h) for j in range(len(x))])
        np.testing.assert_allclose(fd,g,rtol=1e-5,atol=1e-7)
        for b in range(len(self.L)):
            z=x[2*b:2*b+2];_,J=self.con(b,z)
            fd=np.column_stack([(self.con(b,z+np.eye(2)[j]*h)[0]-self.con(b,z-np.eye(2)[j]*h)[0])/(2*h) for j in range(2)])
            np.testing.assert_allclose(fd,J,rtol=1e-5,atol=1e-7)
    def test_projection_feasibility_variational_inequality(self):
        proj=bcd.Projector(self.con);rng=np.random.default_rng(113)
        for b in range(len(self.L)):
            y=rng.normal(size=2)*3;p=proj(b,y,np.zeros(2))
            self.assertLessEqual(self.con(b,p)[0].max(),1e-7)
            # Projection optimality for independently sampled feasible points.
            for _ in range(100):
                z=rng.uniform(-1,1,2)
                if self.con(b,z)[0].max()<=0:self.assertLessEqual((y-p)@(z-p),2e-6)
    def test_safe_step_descent(self):
        x=self.x.copy();proj=bcd.Projector(self.con)
        for k in range(40):
            f,g=self.obj(x);b=k%len(self.L);sl=slice(2*b,2*b+2)
            x[sl]=proj(b,x[sl]-g[sl]/self.L[b],x[sl])
            self.assertLessEqual(self.obj(x)[0],f+1e-9)
    def test_reproducibility_and_cost(self):
        for method in ['cyclic','random','gs','signature','shuffled']:
            a=bcd.solve(self.obj,self.con,self.L,self.x,method,.25,431,maxupdates=20)
            b=bcd.solve(self.obj,self.con,self.L,self.x,method,.25,431,maxupdates=20)
            np.testing.assert_array_equal(a['x'],b['x'])
            self.assertEqual(a['projection_calls'],b['projection_calls'])
            self.assertEqual(a['projection_failures'],0)
            self.assertLessEqual(a['objective'],self.obj(self.x)[0]+1e-9)
            self.assertGreaterEqual(a['projection_calls'],a['updates'])

if __name__=='__main__':unittest.main()
