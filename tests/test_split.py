import unittest
from rail_validator.split import holdout_ids

class SplitTests(unittest.TestCase):
    def test_reproducible_grouped(self):
        rows = [{'app_id':str(i),'split':'train'} for i in range(10)]
        rows += [rows[0], {'app_id':'test-only','split':'test'}]
        a,b = holdout_ids(rows)
        self.assertEqual(len(b),2)
        self.assertFalse(a & b)
        self.assertNotIn('test-only',a|b)
        self.assertEqual((a,b),holdout_ids(reversed(rows)))
