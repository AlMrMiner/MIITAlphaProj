import unittest
from rail_validator.engine import Finding
from rail_validator.ml import correct_candidate, features, grouped_calibration_key
from rail_validator.data import FIELDS

class TestML(unittest.TestCase):
    def setUp(self):
        self.row={field:'0' for field in FIELDS}
        self.row.update(app_id='Z1',split='train',channel='email',created_at='2025-01-01',
                        departure_station_code='123456',destination_station_code='654321',
                        weight_t='120',wagons_count='2')
    def test_features_never_contain_truth_or_identifiers(self):
        feature=features(self.row,Finding('cargo_name','Название', 'Нефть','name_from_fuzzy'))
        for key in ('correct_value','error_type','app_id','split'):
            self.assertNotIn(key,feature)
    def test_candidate_truth(self):
        labels=[{'field':'weight_t','correct_value':'120.0'}]
        self.assertEqual(correct_candidate('weight_t','120',labels),1)
        self.assertEqual(correct_candidate('weight_t','110',labels),0)
        self.assertEqual(correct_candidate('cargo_code','123',labels),0)
    def test_calibration_group_deterministic(self):
        self.assertEqual(grouped_calibration_key(self.row),grouped_calibration_key(dict(self.row)))

if __name__=='__main__':
    unittest.main()
