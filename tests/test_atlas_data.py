"""Fast, dependency-free regression tests for the authored atlas dataset."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.validate_atlas import DEFAULT_PATH, load_atlas, validate


class AtlasDataTests(unittest.TestCase):
    def setUp(self):
        self.data = load_atlas()

    def test_valid_dataset(self):
        validate(self.data)

    def test_all_sources_are_referenced(self):
        used = {s for collection in ('milestones', 'capabilities', 'patterns') for row in self.data[collection] for s in row['sources']}
        self.assertEqual(used, {s['id'] for s in self.data['sources']})

    def test_unique_source_urls(self):
        urls = [s['url'] for s in self.data['sources']]
        self.assertEqual(len(urls), len(set(urls)))

    def test_reject_duplicate_ids(self):
        self.data['milestones'].append(copy.deepcopy(self.data['milestones'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate(self.data)

    def test_reject_dangling_source(self):
        self.data['milestones'][0]['sources'] = ['does-not-exist']
        with self.assertRaisesRegex(ValueError, 'Unknown sources'):
            validate(self.data)

    def test_reject_unsafe_url(self):
        self.data['sources'][0]['url'] = 'javascript:alert(1)'
        with self.assertRaisesRegex(ValueError, 'Unsafe source'):
            validate(self.data)

    def test_reject_future_claim(self):
        self.data['milestones'][0]['date'] = '2099-01-01'
        with self.assertRaisesRegex(ValueError, 'outside declared coverage'):
            validate(self.data)

    def test_reject_untyped_date(self):
        self.data['milestones'][0]['date_type'] = 'Release'
        with self.assertRaisesRegex(ValueError, 'date meaning'):
            validate(self.data)

    def test_reject_missing_caveat(self):
        self.data['milestones'][0]['limit'] = ''
        with self.assertRaisesRegex(ValueError, 'Unqualified claim'):
            validate(self.data)

    def test_reject_invalid_activity_reference(self):
        self.data['milestones'][0]['activities'] = ['nonexistent']
        with self.assertRaisesRegex(ValueError, 'Unknown activities'):
            validate(self.data)

    def test_reject_capability_type_confusion(self):
        self.data['capabilities'][0]['kind'] = 'product'
        with self.assertRaisesRegex(ValueError, 'Unknown capability'):
            validate(self.data)

    def test_reject_changed_cardinality(self):
        self.data['patterns'].pop()
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            validate(self.data)

    def test_standalone_html_has_no_runtime_network_dependencies(self):
        html = DEFAULT_PATH.read_text()
        self.assertNotIn('fetch(', html)
        self.assertNotIn('@import', html)
        self.assertNotIn('@font-face', html)
        self.assertIn('application/json', html)

    def test_reject_external_script(self):
        html = DEFAULT_PATH.read_text().replace('</head>', '<script src="https://cdn.example/a.js"></script></head>')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.html'
            path.write_text(html)
            with self.assertRaisesRegex(ValueError, 'Runtime resources'):
                load_atlas(path)

    def test_core_historical_distinctions(self):
        items = {m['id']: m for m in self.data['milestones']}
        self.assertEqual(items['gpt4']['date'], '2023-03-15')
        self.assertEqual(items['instruct']['date_type'], 'Paper submission')
        self.assertEqual(items['mcp']['kind'], 'protocol')
        self.assertEqual(items['chatgpt']['kind'], 'product')
        self.assertIn('encoder', items['bert']['limit'])


if __name__ == '__main__':
    unittest.main()
