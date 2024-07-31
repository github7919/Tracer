# python -m unittest tests.test_reference

import unittest
import time
import json
from pytrace.reference import ReferenceTracker
from pytrace.utils import can_weakref

class TestReferenceTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ReferenceTracker()
        self.obj = {}
        self.obj_name = 'test_obj'

    def test_track_object(self):
        self.tracker.track(self.obj, self.obj_name)
        tracked_objects = self.tracker.list_tracked_objects()
        self.assertIn(self.obj_name, tracked_objects)
        self.assertEqual(tracked_objects[self.obj_name], self.obj)

    def test_reference_count(self):
        self.tracker.track(self.obj, self.obj_name)
        ref_count = self.tracker.reference_count(self.obj_name)
        self.assertGreater(ref_count, 0)

    def test_get_lifetime(self):
        self.tracker.track(self.obj, self.obj_name)
        time.sleep(1)
        lifetime = self.tracker.get_lifetime(self.obj_name)
        self.assertIsNotNone(lifetime)
        self.assertGreaterEqual(lifetime, 1)

    def test_clear(self):
        self.tracker.track(self.obj, self.obj_name)
        self.tracker.clear()
        tracked_objects = self.tracker.list_tracked_objects()
        self.assertNotIn(self.obj_name, tracked_objects)

    def test_to_json(self):
        self.tracker.track(self.obj, self.obj_name)
        json_data = self.tracker.to_json()
        data = json.loads(json_data)
        self.assertIn(self.obj_name, data)

    def test_alert_on_threshold(self):
        self.tracker.track(self.obj, self.obj_name)
        alerts = self.tracker.alert_on_threshold(ref_count_threshold=10, memory_threshold=10)
        self.assertTrue(len(alerts) > 0)

    def test_track_lifetime_changes(self):
        changes = self.tracker.track_lifetime_changes(self.obj, interval=0.1, duration=1)
        self.assertTrue(len(changes) > 0)

    def test_generate_memory_profile_report(self):
        try:
            self.tracker.track(self.obj, self.obj_name)
            self.tracker.generate_memory_profile_report('test_report.txt')
            # Assuming the report generation doesn't raise an exception, the test passes
        except Exception as e:
            self.fail(f"generate_memory_profile_report raised an exception: {e}")

    def test_get_reference_chain(self):
        chain = self.tracker.get_reference_chain(self.obj, max_depth=2)
        self.assertIsInstance(chain, list)

    def test_get_object_metadata(self):
        self.tracker.track(self.obj, self.obj_name)
        metadata = self.tracker.get_object_metadata(self.obj_name)
        self.assertIn('type', metadata)
        self.assertIn('id', metadata)

if __name__ == '__main__':
    unittest.main()
