import csv
import json
import tempfile
import unittest
from pathlib import Path
from rail_validator.web import Workspace, handler_for
from rail_validator.process import AUDIT_FIELDS, REVIEW_FIELDS


class WebWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'metrics').mkdir()
        self.ws = Workspace(self.root)
        self.source = self.ws.output/'demo-g3-holdout'
        self.source.mkdir()
        item = {'app_id': 'Z0001', 'field':'planned_date','original':'2024-12-01',
                'candidate':'','reason':'Дата в прошлом','rule':'old_date','confidence':'0', 'action':'review'}
        for filename, fields, rows in [
            ('queue.csv', AUDIT_FIELDS, [item]),
            ('operator_decisions.csv', REVIEW_FIELDS, [{**item,'operator_action':'','operator_value':''}]),
        ]:
            with (self.source/filename).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader();writer.writerows(rows)

    def test_queue_hides_confidence_for_missing_candidate(self):
        q=self.ws.queue('demo-g3-holdout')
        self.assertEqual(q['total'],1)
        self.assertEqual(q['rows'][0]['confidence_display'],'—')

    def test_edit_decision_persists(self):
        self.ws.save_decision('demo-g3-holdout','Z0001','planned_date','edit','2025-01-01')
        q=self.ws.queue('demo-g3-holdout')
        self.assertEqual(q['rows'][0]['operator_action'],'edit')
        self.assertEqual(q['rows'][0]['operator_value'],'2025-01-01')

    def test_cannot_accept_missing_candidate(self):
        with self.assertRaisesRegex(ValueError, 'отсутствует'):
            self.ws.save_decision('demo-g3-holdout','Z0001','planned_date','accept','')

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            self.ws.queue('../invalid')

    def test_fixed_process_arguments(self):
        (self.root/'metrics'/'confidence.json').write_text('{}',encoding='utf-8')
        import unittest.mock as mock
        with mock.patch('threading.Thread') as thread:
            self.ws.new_job('process',{'method':'g3','split':'holdout'})
            argv=thread.call_args.kwargs['args'][1]
            self.assertEqual(argv[-2:], ['--confidence',str(self.root/'metrics'/'confidence.json')])
            self.assertIn(str(self.ws.output/'demo-g3-holdout'), argv)

if __name__ == '__main__':unittest.main()
