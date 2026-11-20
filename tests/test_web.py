import csv
import json
import os
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

    def test_ticket_displays_standard_date(self):
        from rail_validator.data import FIELDS
        record={f:'x' for f in FIELDS}
        record.update(app_id='Z0001',planned_date='45672',created_at='2025-01-03T10:00:00')
        with (self.source/'pending.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerow(record)
        t=self.ws.ticket('demo-g3-holdout','Z0001')
        d={f['name']:f for f in t['fields']}
        from rail_validator.validation import parse_date
        self.assertEqual(d['planned_date']['display'],parse_date('45672').strftime('%d.%m.%Y'))
        self.assertEqual(d['created_at']['display'],'03.01.2025 10:00')
        self.ws.save_ticket_field('demo-g3-holdout','Z0001','planned_date','25.01.2025')
        self.assertEqual(self.ws.queue('demo-g3-holdout')['rows'][0]['operator_value'],'2025-01-25')

    def test_ticket_changes_nonflagged_field_in_resolve(self):
        from rail_validator.data import FIELDS
        from rail_validator.process import resolve
        record={f:'x' for f in FIELDS}
        record.update(app_id='Z0001',planned_date='2024-12-01',cargo_name='Старый груз')
        for filename,rows in [('pending.csv',[record]),('ready.csv',[])]:
            with (self.source/filename).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        self.ws.save_ticket_field('demo-g3-holdout','Z0001','cargo_name','Новый груз')
        self.ws.save_decision('demo-g3-holdout','Z0001','planned_date','edit','2025-01-25')
        resolve(self.source,self.source/'operator_decisions.csv',self.ws.output/'resolved-test')
        with (self.ws.output/'resolved-test'/'approved.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(rows[0]['cargo_name'],'Новый груз')
        self.assertEqual(rows[0]['planned_date'],'2025-01-25')
        with (self.ws.output/'resolved-test'/'operator_audit.csv').open(encoding='utf-8',newline='') as f:
            self.assertEqual(len(list(csv.DictReader(f))),2)

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
            self.assertEqual(argv[-2], '--confidence')
            self.assertTrue(
                os.path.samefile(argv[-1], self.root / 'metrics' / 'confidence.json'),
                f'Путь к confidence.json неверен: {argv[-1]}',
            )
            self.assertIn('--output', argv)
            output_path = Path(argv[argv.index('--output') + 1])
            self.assertEqual(output_path.name, 'demo-g3-holdout')
            self.assertTrue(os.path.samefile(output_path.parent, self.ws.output))

if __name__ == '__main__':unittest.main()
