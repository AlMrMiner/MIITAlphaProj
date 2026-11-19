import unittest
from rail_validator.validation import inn_valid, parse_date, local_checks

class ValidationTests(unittest.TestCase):
    def test_inn(self):
        self.assertTrue(inn_valid('7707083893'))
        self.assertFalse(inn_valid('7707083894'))
    def test_dates(self):
        self.assertEqual(parse_date('05.01.2025'),parse_date('2025-01-05'))
        self.assertIsNone(parse_date('31.02.2025'))
    def test_queue_multiple(self):
        row = dict(client_inn='1',consignee_inn='2',departure_station_code='a',destination_station_code='123456',created_at='2025-01-02 10:00',planned_date='01.01.2025',weight_t='NaN',wagons_count='2.5')
        self.assertEqual(len(local_checks(row)),6)
