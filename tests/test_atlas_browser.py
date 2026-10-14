"""Optional real-browser regression tests; pip install playwright first.

Run: python -m unittest discover -s tests -p 'test_atlas_browser.py' -v
Set CHROMIUM_EXECUTABLE to an installed browser, or install Playwright Chromium.
The HTML is rendered from memory with all runtime network requests blocked.
"""
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / 'docs/index.html').read_text(encoding='utf-8')


@unittest.skipIf(sync_playwright is None, 'Optional dependency: playwright')
class AtlasBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        executable = os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium')
        options = {'headless': True}
        if executable:
            options['executable_path'] = executable
        cls.browser = cls.pw.chromium.launch(**options)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1680, 'height': 1050})
        self.blocked = []
        self.errors = []
        def serve(route):
            self.blocked.append(route.request.url)
            route.abort()
        self.context.route('**/*', serve)
        self.page = self.context.new_page()
        self.page.on('pageerror', lambda err: self.errors.append(str(err)))
        self.page.set_content(HTML, wait_until='load')

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], 'Uncaught browser errors')
        self.assertEqual(self.blocked, [], 'Unexpected runtime network dependency')

    def view(self, name):
        self.page.locator(f'[data-view="{name}"]').click()

    def test_initial_map(self):
        self.assertEqual(self.page.locator('[data-select-type="activity"]').count(), 6)
        self.assertEqual(self.page.locator('[data-select-type="capability"]').count(), 12)
        self.assertEqual(self.page.locator('h1').count(), 1)

    def test_each_activity_has_work_product_and_check(self):
        for name in ['find','understand','analyze','create','evaluate','communicate']:
            self.page.locator(f'[data-filter="{name}"]').click()
            panel = self.page.locator('#inspector').inner_text()
            self.assertIn('WORK PRODUCT', panel)
            self.assertIn('YOUR CHECK', panel)
            self.assertEqual(self.page.evaluate('state.activity'), name)

    def test_capability_drilldown(self):
        for name in ['search','retrieval','longcontext','multimodal','reasoning','execution','generation','connectors','verification','artifacts']:
            self.page.locator(f'[data-select-type="capability"][data-select-id="{name}"]').first.click()
            self.assertGreater(self.page.locator('#inspector .source-link').count(), 0)
            self.assertIn('important distinction', self.page.locator('#inspector').inner_text().lower())

    def test_scientific_and_general_examples(self):
        self.page.locator('[data-filter="find"]').click()
        self.assertIn('enzyme-function', self.page.locator('#inspector').inner_text())
        self.page.select_option('#context', 'general')
        self.assertIn('decision', self.page.locator('#inspector').inner_text())
        self.assertIn('Source brief', self.page.locator('#figure').inner_text())

    def test_search_type_and_empty_state(self):
        self.view('history')
        self.assertEqual(self.page.locator('[data-select-type="milestone"]').count(), 16)
        self.page.locator('#search').fill('ReAct')
        self.assertEqual(self.page.locator('[data-select-type="milestone"]').count(), 1)
        self.page.select_option('#kind', 'protocol')
        self.assertIn('No matching milestones', self.page.locator('#figure').inner_text())
        self.page.locator('#reset').click()
        self.assertEqual(self.page.locator('[data-select-type="milestone"]').count(), 16)

    def test_all_milestones_have_date_meaning_and_sources(self):
        self.view('history')
        ids = self.page.evaluate('DATA.milestones.map(m=>m.id)')
        for name in ids:
            self.page.locator(f'[data-select-type="milestone"][data-select-id="{name}"]').click()
            self.assertGreater(self.page.locator('#inspector .source-link').count(), 0)
            self.assertIn('what this does not establish', self.page.locator('#inspector').inner_text().lower())

    def test_patterns(self):
        self.view('patterns')
        self.assertEqual(self.page.locator('[data-select-type="pattern"]').count(), 5)
        for name in ['completion','conversation','workflow','agent','multiagent']:
            self.page.locator(f'[data-select-id="{name}"]').click()
            self.assertIn('YOUR CHECK', self.page.locator('#inspector').inner_text())

    def test_keyboard_tabs(self):
        self.page.locator('#tab-map').focus()
        self.page.keyboard.press('ArrowRight')
        self.assertEqual(self.page.locator('#tab-history').get_attribute('aria-selected'), 'true')
        self.page.keyboard.press('End')
        self.assertEqual(self.page.locator('#tab-patterns').get_attribute('aria-selected'), 'true')
        self.page.keyboard.press('Home')
        self.assertEqual(self.page.locator('#tab-map').get_attribute('aria-selected'), 'true')

    def test_keyboard_diagram(self):
        target = self.page.locator('[data-select-type="activity"][data-select-id="analyze"]')
        target.focus()
        self.page.keyboard.press('Enter')
        self.assertEqual(self.page.locator('#inspector h2').inner_text(), 'Analyze')
        self.assertEqual(self.page.evaluate('document.activeElement.dataset.selectId'), 'analyze')

    def test_sources_modal(self):
        self.page.locator('#sources-open').click()
        self.assertTrue(self.page.locator('#sources-dialog').is_visible())
        self.assertEqual(self.page.locator('#source-list li').count(), 16)
        self.page.keyboard.press('Escape')
        self.assertFalse(self.page.locator('#sources-dialog').is_visible())

    def test_presentation_keyboard(self):
        self.page.keyboard.press('f')
        self.assertIn('present', self.page.locator('body').get_attribute('class'))
        self.page.keyboard.press('Escape')
        self.assertNotIn('present', self.page.locator('body').get_attribute('class'))

    def test_mobile_scroll_is_contained(self):
        self.page.set_viewport_size({'width': 390, 'height': 844})
        for name in ['map','history','patterns']:
            self.view(name)
            self.assertFalse(self.page.evaluate('document.documentElement.scrollWidth > innerWidth'))
            self.assertTrue(self.page.evaluate('document.querySelector("#figure").scrollWidth > document.querySelector("#figure").clientWidth'))

    def test_deep_link_parser(self):
        self.page.evaluate('loadHash("#view=map&activity=analyze&context=general&selected=activity%3Aanalyze"); render();')
        self.assertEqual(self.page.evaluate('state.activity'), 'analyze')
        self.assertEqual(self.page.evaluate('state.context'), 'general')
        self.assertEqual(self.page.locator('#inspector h2').inner_text(), 'Analyze')

    def test_search_content_cannot_inject_markup(self):
        self.view('history')
        self.page.locator('#search').fill('\"><img src=x onerror=alert(1)>')
        self.assertEqual(self.page.locator('#figure img').count(), 0)
        self.assertIn('No matching', self.page.locator('#figure').inner_text())

    def test_text_fits_svg_and_cards(self):
        for name in ['map','history','patterns']:
            self.view(name)
            overflow = self.page.evaluate('''() => {
              const svg = document.querySelector('#figure svg');
              const vb = svg.viewBox.baseVal;
              return [...svg.querySelectorAll('text')].filter(t => {
                const b=t.getBBox();
                if(b.x < -1 || b.y < -1 || b.x+b.width > vb.width+1 || b.y+b.height > vb.height+1) return true;
                const g=t.closest('[data-select-type]');
                const r=g?.querySelector('.hit');
                if(!r) return false;
                const q=r.getBBox();
                return b.x < q.x-1 || b.y < q.y-1 || b.x+b.width > q.x+q.width+1 || b.y+b.height > q.y+q.height+1;
              }).map(t=>t.textContent);
            }''')
            self.assertEqual(overflow, [], name)

    def test_svg_and_png_export_for_every_view(self):
        for name in ['map','history','patterns']:
            self.view(name)
            with self.page.expect_download() as got:
                self.page.locator('#export-svg').click()
            svg_path = Path(got.value.path())
            root = ET.fromstring(svg_path.read_text())
            meta = json.loads(root.find('{http://www.w3.org/2000/svg}metadata').text)
            self.assertEqual(meta['view'], name)
            self.assertEqual(len(meta['sources']), 16)
            with self.page.expect_download() as got:
                self.page.locator('#export-png').click()
            payload = Path(got.value.path()).read_bytes()
            self.assertEqual(payload[:8], b'\x89PNG\r\n\x1a\n')
            width, height = struct.unpack('>II', payload[16:24])
            self.assertEqual(width, 2720)
            self.assertGreater(height, 1000)


if __name__ == '__main__':
    unittest.main()
