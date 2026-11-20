import unittest
from rail_validator.engine import Engine, Finding, fingerprint, wilson, equivalent
from rail_validator.validation import station_valid, parse_date
from rail_validator.reference import LAYOUT
from rail_validator.evaluate import partition

class EngineTests(unittest.TestCase):
    def test_station_checksum(self):
        self.assertTrue(station_valid('488759'))
        self.assertFalse(station_valid('488750'))
    def test_excel_and_short_dates(self):
        self.assertEqual(parse_date('45658'),parse_date('01.01.25'))
    def test_layout(self):
        self.assertEqual('ytanm'.translate(LAYOUT),'нефть')
    def test_threshold(self):
        e=Engine(None,True)
        f={'cargo_name':Finding('cargo_name','reason','Кирпич','rule')}
        e.confidence={'rule':.949}
        self.assertEqual(e.decisions(f)['cargo_name']['action'],'review')
        e.confidence={'rule':.95}
        self.assertEqual(e.decisions(f)['cargo_name']['action'],'auto')
        self.assertLess(wilson(1,1),.95)
        self.assertGreater(wilson(100,100),.95)
    def test_no_candidate_has_no_confidence(self):
        engine=Engine(None,True)
        engine.confidence={'kg_to_t':.995}
        findings={'weight_t':Finding('weight_t','Масса превышает грузоподъёмность',None,'kg_to_t')}
        decision=engine.decisions(findings)['weight_t']
        self.assertIsNone(decision['confidence'])
        self.assertEqual(decision['action'],'review')

    def test_group_partition(self):
        rows=[dict(app_id=str(i),split='train',created_at=str(i),channel='API',planned_date='2025-01-01',cargo_name=str(i//2)) for i in range(20)]
        hold=partition(rows)
        self.assertEqual(hold,partition(rows))
        for i in range(0,20,2):self.assertEqual(str(i) in hold,str(i+1) in hold)
    def test_equivalence(self):
        self.assertTrue(equivalent('weight_t','150.0','150'))
        self.assertFalse(equivalent('cargo_name','уголь','Уголь'))
