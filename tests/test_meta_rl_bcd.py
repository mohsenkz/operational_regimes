import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'experiments/meta_rl_bcd/benchmark_meta.py'
spec=importlib.util.spec_from_file_location('meta',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestMeta(unittest.TestCase):
 def test_clone_is_independent_and_resets_optimizer(self):
  p=m.rl.Policy(17);p.steps=7;p.m['W'][:]=1
  q=m.clone(p)
  for k in p.p:np.testing.assert_array_equal(q.p[k],p.p[k])
  self.assertEqual(q.steps,0);self.assertTrue(np.all(q.m['W']==0))
  q.p['W'][0,0]+=1;self.assertNotEqual(q.p['W'][0,0],p.p['W'][0,0])
 def test_reptile_exact_update(self):
  p=m.rl.Policy(18);q=m.clone(p);old={k:v.copy() for k,v in p.p.items()}
  for k in q.p:q.p[k]+=2
  m.interpolate(p,q,.25)
  for k in p.p:np.testing.assert_allclose(p.p[k],old[k]+.5)
 def test_inner_update_and_query_freeze(self):
  p=m.rl.Policy(19);q=m.clone(p);cfg=m.TRAIN[0]
  m.rl.episode(51000001,cfg,'base',q,52000001,limit=12,train=True)
  self.assertGreater(np.linalg.norm(q.p['W']-p.p['W']),0)
  old={k:v.copy() for k,v in q.p.items()}
  m.rl.episode(51000002,cfg,'base',q,52000002,limit=12,train=False)
  for k in q.p:np.testing.assert_array_equal(old[k],q.p[k])
 def test_task_reproducibility_and_family_holdout(self):
  self.assertEqual(m.task(m.TRAIN[0],99),m.task(m.TRAIN[0],99))
  train={(c['blocks'],c['per_block'],c['frequency']) for c in m.TRAIN}
  self.assertTrue(all((c['blocks'],c['per_block'],c['frequency']) not in train for c in m.TEST))
if __name__=='__main__':unittest.main()
