# Tracer/reference.py
# python -m Tracer.reference

import weakref
import gc
import sys
import csv
import time
import json
import threading
from pympler import asizeof
from fpdf import FPDF
import pandas as pd
from graphviz import Digraph
from collections import defaultdict
from Tracer.utils import can_weakref

class ReferenceTracker:
    """
    A class for tracking and analyzing object references in Python.

    This class provides comprehensive tools for monitoring object lifetimes,
    reference counts, memory usage, and detecting potential memory leaks.

    Methods:
    - __init__(): Initialize the ReferenceTracker.
    - track(obj, obj_name=None): Track a new object.
    - list_tracked_objects(): List all currently tracked objects.
    - reference_count(obj_name): Get the reference count for a tracked object.
    - get_lifetime(obj_name): Get the lifetime of a tracked object.
    - clear(): Clear all tracked objects.
    - to_json(): Export tracked object data to JSON format.
    - track_object_changes(obj, interval=1, duration=10): Track changes in an object over time.
    - detect_cycles(): Detect reference cycles in tracked objects.
    - get_object_metadata(obj_name): Get metadata for a tracked object.
    - alert_on_threshold(memory_threshold=None, refcount_threshold=None): Set alerts for memory usage or reference count thresholds.
    - get_reference_details(obj): Get detailed reference information for an object.
    - track_lifetime_changes(obj, interval=1, duration=10): Track changes in an object's lifetime.
    - generate_memory_profile_report(file_path, format='txt'): Generate a memory profile report.
    - _update_history(obj_name, obj): Update history of reference count and memory usage for an object.
    """

    def __init__(self):
        """
        Initialize the ReferenceTracker with necessary data structures.
        """
        self._references = {}  # Strong references to objects
        self._weak_references = weakref.WeakValueDictionary()  # Weak references to objects
        self._creation_times = {}  # Tracks when objects were created
        self._destruction_times = {}  # Tracks when objects were destroyed
        self._ref_count_history = defaultdict(list)  # History of reference counts
        self._memory_usage_history = defaultdict(list)  # History of memory usage

    def track(self, obj, obj_name=None):
        """
        Track a new object.

        Args:
            obj: The object to track.
            obj_name: Optional name for the object. If not provided, a name is generated.

        This method adds the object to the tracker and initializes its metadata.
        """
        obj_name = obj_name or f"obj_{id(obj)}"
        try:
            if can_weakref(obj):
                self._weak_references[obj_name] = obj
            else:
                self._references[obj_name] = obj
            self._creation_times[obj_name] = time.time()
            self._destruction_times[obj_name] = None
            self._update_history(obj_name, obj)
        except Exception as e:
            print(f"Error tracking object {obj_name}: {e}")

    def _update_history(self, obj_name, obj):
        """
        Update the history of reference count and memory usage for an object.

        Args:
            obj_name: The name of the object.
            obj: The object itself.

        This method is called internally to keep track of changes over time.
        """
        current_time = time.time()
        self._ref_count_history[obj_name].append((current_time, sys.getrefcount(obj) - 1))
        self._memory_usage_history[obj_name].append((current_time, asizeof.asizeof(obj)))

    def list_tracked_objects(self):
        """
        List all currently tracked objects.

        Returns:
            A dictionary of tracked objects, including both weakly and strongly referenced objects.
        """
        tracked_objects = {name: obj for name, obj in self._weak_references.items() if obj is not None}
        tracked_objects.update(self._references)
        return tracked_objects

    def reference_count(self, obj_name):
        """
        Get the reference count for a tracked object.

        Args:
            obj_name: The name of the object.

        Returns:
            The reference count of the object, or 0 if the object is not found.
        """
        obj = self._weak_references.get(obj_name) or self._references.get(obj_name)
        if obj is None:
            return 0  # If the object is not found, return 0 reference count
        return sys.getrefcount(obj) - 1  # Subtract one to account for the reference created by getrefcount itself

    def get_lifetime(self, obj_name):
        """
        Get the lifetime of a tracked object.

        Args:
            obj_name: The name of the object.

        Returns:
            The lifetime of the object in seconds, or None if the creation time is not found.
        """
        creation_time = self._creation_times.get(obj_name)
        if creation_time is None:
            print(f"Warning: Creation time for object '{obj_name}' not found.")
            return None
        
        destruction_time = self._destruction_times.get(obj_name, time.time())
        return destruction_time - creation_time

    def clear(self):
        """
        Clear all tracked objects and record their destruction times.
        """
        current_time = time.time()
        for name in list(self._references.keys()) + list(self._weak_references.keys()):
            self._destruction_times[name] = current_time
        self._references.clear()
        self._weak_references.clear()

    def to_json(self):
        """
        Export tracked object data to JSON format.

        Returns:
            A JSON string containing data about all tracked objects.
        """
        tracked_data = {}
        for name, obj in self.list_tracked_objects().items():
            lifetime = self.get_lifetime(name)
            if lifetime is None:
                continue
            tracked_data[name] = {
                'type': type(obj).__name__,
                'reference_count': self.reference_count(name),
                'lifetime': lifetime,
                'memory_usage': asizeof.asizeof(obj),
                'ref_count_history': self._ref_count_history[name],
                'memory_usage_history': self._memory_usage_history[name]
            }
        return json.dumps(tracked_data, indent=2, default=str)

    def visualize_references(self):
        """
        Visualizes the references of tracked objects using Graphviz.
        
        Returns:
        - A graphviz.Digraph object representing the reference graph.

        This method creates a visual representation of object references,
        which can be useful for understanding complex object relationships.
        """
        dot = Digraph(comment='Object Reference Graph')
        
        for name, obj in self.list_tracked_objects().items():
            label = (f'{name}\n{type(obj).__name__}\n'
                     f'Refs: {self.reference_count(name)}\n'
                     f'Lifetime: {self.get_lifetime(name):.2f}s\n'
                     f'Memory: {asizeof.asizeof(obj)} bytes')
            dot.node(name, label=label)
            
            obj_id = id(obj)
            for ref_name, ref_obj in self.list_tracked_objects().items():
                if obj is not ref_obj and obj_id in map(id, gc.get_referrers(ref_obj)):
                    dot.edge(ref_name, name)
        
        return dot

    def track_object_changes(self, obj, interval=1.0, duration=10.0):
        """
        Tracks the changes in reference count and memory usage of an object over time.
        
        Parameters:
        - obj: The object to track.
        - interval: Time in seconds between checks (default: 1.0).
        - duration: Total duration for tracking in seconds (default: 10.0).
        
        Returns:
        - A list of dictionaries recording changes over time.

        This method is useful for monitoring how an object's properties change
        during program execution, which can help identify memory leaks or
        unexpected behavior.
        """
        obj_name = str(id(obj))
        changes = []
        start_time = time.time()
        stop_event = threading.Event()

        def track():
            while not stop_event.is_set():
                current_time = time.time() - start_time
                if current_time >= duration:
                    break
                try:
                    ref_count = sys.getrefcount(obj) - 2  # Subtract 2 for getrefcount and this function's reference
                    memory_usage = asizeof.asizeof(obj)
                    changes.append({
                        'time': current_time,
                        'ref_count': ref_count,
                        'memory_usage': memory_usage,
                    })
                except Exception as e:
                    print(f"Error tracking object {obj_name}: {e}")
                time.sleep(interval)

        tracking_thread = threading.Thread(target=track)
        tracking_thread.start()

        try:
            tracking_thread.join(timeout=duration)
        finally:
            stop_event.set()

        return changes

    def detect_cycles(self):
        """
        Detects reference cycles in tracked objects.

        Returns:
        - A list of objects involved in reference cycles.

        This method uses a depth-first search algorithm to detect cycles
        in the object reference graph. It's useful for identifying potential
        memory leaks caused by circular references.
        """
        def visit(node, visited, stack):
            visited[node] = True
            stack[node] = True

            for neighbor in gc.get_referents(node):
                if isinstance(neighbor, dict) or isinstance(neighbor, (list, tuple, set)):
                    continue  # Skip dictionaries and collections to avoid false positives
                if not visited.get(neighbor):
                    if visit(neighbor, visited, stack):
                        return True
                elif stack.get(neighbor):
                    return True

            stack[node] = False
            return False

        visited = defaultdict(bool)
        stack = defaultdict(bool)
        cycles = []

        for obj in self.list_tracked_objects().values():
            if not visited[obj]:
                if visit(obj, visited, stack):
                    cycles.append(obj)  # Adjust to collect all objects involved in the cycle

        return cycles

    def get_object_metadata(self, obj_name):
        """
        Retrieves detailed metadata about a tracked object.
        
        Parameters:
        - obj_name: The name of the object to get metadata for.
        
        Returns:
        - A dictionary containing metadata such as type, reference count, memory usage, etc.

        This method provides a comprehensive overview of an object's current state,
        which is useful for debugging and performance analysis.
        """
        obj = self._weak_references.get(obj_name) or self._references.get(obj_name)
        if obj is None:
            return {}
        return {
            'type': type(obj).__name__,
            'id': id(obj),
            'reference_count': self.reference_count(obj_name),
            'memory_usage': asizeof.asizeof(obj),
            'creation_time': self._creation_times.get(obj_name),
            'lifetime': self.get_lifetime(obj_name),
        }

    def export_to_csv(self, file_path):
        """
        Exports the tracked objects and their metadata to a CSV file.
        
        Parameters:
        - file_path: The path to the file where the data will be saved.

        This method is useful for generating reports or preparing data
        for further analysis in spreadsheet software.
        """
        fieldnames = ['name', 'type', 'reference_count', 'memory_usage', 'creation_time', 'lifetime', 'is_weakly_referenced']
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                current_time = time.time()
                for name, obj in self.list_tracked_objects().items():
                    is_weak = name in self._weak_references
                    writer.writerow({
                        'name': name,
                        'type': type(obj).__name__,
                        'reference_count': self.reference_count(name),
                        'memory_usage': asizeof.asizeof(obj),
                        'creation_time': self._creation_times.get(name, 'Unknown'),
                        'lifetime': current_time - self._creation_times.get(name, current_time),
                        'is_weakly_referenced': is_weak
                    })
            print(f"Data exported successfully to {file_path}")
        except IOError as e:
            print(f"Error writing to file {file_path}: {e}")
        except Exception as e:
            print(f"An error occurred while exporting data: {e}")

    def alert_on_threshold(self, ref_count_threshold=None, memory_threshold=None):
        """
        Alerts if any tracked object exceeds the specified reference count or memory usage thresholds.
        
        Parameters:
        - ref_count_threshold: The reference count threshold.
        - memory_threshold: The memory usage threshold in bytes.
        
        Returns:
        - A list of alerts for objects exceeding the thresholds.

        This method is useful for identifying objects that may be consuming
        excessive resources or not being properly garbage collected.
        """
        alerts = []
        for name, obj in self.list_tracked_objects().items():
            try:
                ref_count = self.reference_count(name)
                memory_usage = asizeof.asizeof(obj)
                if (ref_count_threshold is not None and ref_count > ref_count_threshold) or \
                   (memory_threshold is not None and memory_usage > memory_threshold):
                    alerts.append({
                        'name': name,
                        'type': type(obj).__name__,
                        'reference_count': ref_count,
                        'memory_usage': memory_usage,
                    })
            except Exception as e:
                print(f"Error processing object {name}: {e}")

        if not alerts:
            # Provide a default message if no alerts were generated
            alerts.append("No objects exceeded the specified thresholds.")

        return alerts

    def get_reference_details(self, obj):
        """
        Retrieves details about which objects are referencing the given object.
        
        Parameters:
        - obj: The object for which to find references.
        
        Returns:
        - A list of dictionaries containing the id, type, and name (if tracked) of the referencing objects.

        This method is helpful for understanding complex object relationships
        and identifying unexpected references that may prevent garbage collection.
        """
        referrers = gc.get_referrers(obj)
        details = []
        
        for ref in referrers:
            if ref is not obj:
                ref_id = id(ref)
                ref_type = type(ref).__name__
                ref_name = None
                for name, tracked_obj in self.list_tracked_objects().items():
                    if tracked_obj is ref:
                        ref_name = name
                        break
                details.append({
                    'id': ref_id,
                    'type': ref_type,
                    'name': ref_name
                })
        
        return details

    def track_lifetime_changes(self, obj, interval=1.0, duration=10.0):
        """
        Tracks the changes in lifetime of an object over time.
        
        Parameters:
        - obj: The object to track.
        - interval: Time in seconds between checks (default: 1.0).
        - duration: Total duration for tracking in seconds (default: 10.0).
        
        Returns:
        - A list of dictionaries recording changes over time.

        This method is useful for monitoring how long an object persists
        and can help identify objects that are living longer than expected.
        """
        obj_name = str(id(obj))
        changes = []
        start_time = time.time()
        stop_event = threading.Event()

        def track():
            while not stop_event.is_set():
                current_time = time.time() - start_time
                if current_time >= duration:
                    break
                try:
                    lifetime = self.get_lifetime(obj_name)
                    changes.append({
                        'time': current_time,
                        'lifetime': lifetime,
                    })
                except Exception as e:
                    print(f"Error tracking lifetime of object {obj_name}: {e}")
                time.sleep(interval)

        tracking_thread = threading.Thread(target=track)
        tracking_thread.start()

        try:
            tracking_thread.join(timeout=duration)
        finally:
            stop_event.set()

        return changes

    def generate_memory_profile_report(self, file_path, format='txt'):
        """
        Generates a detailed report on memory usage and reference counts,
        saved to a specified file path in the requested format.
        
        Parameters:
        - file_path: The path to the file where the report will be saved.
        - format: The format of the file ('txt', 'csv', 'html', 'pdf', 'xlsx').
        """
        format = format.lower()
        report_data = self._gather_report_data()
        
        if format == 'txt':
            self._generate_txt_report(file_path, report_data)
        elif format == 'csv':
            self._generate_csv_report(file_path, report_data)
        elif format == 'html':
            self._generate_html_report(file_path, report_data)
        elif format == 'pdf':
            self._generate_pdf_report(file_path, report_data)
        elif format == 'xlsx':
            self._generate_xlsx_report(file_path, report_data)
        else:
            raise ValueError("Unsupported format. Choose from 'txt', 'csv', 'html', 'pdf', 'xlsx'.")

    def _gather_report_data(self):
        report_data = []
        for name, obj in self.list_tracked_objects().items():
            lifetime = self.get_lifetime(name)
            if lifetime is None:
                continue
            report_data.append({
                'Name': name,
                'Type': type(obj).__name__,
                'Reference Count': self.reference_count(name),
                'Memory Usage': asizeof.asizeof(obj),
                'Creation Time': self._creation_times.get(name),
                'Lifetime': lifetime,
            })
        return report_data

    def _generate_txt_report(self, file_path, report_data):
        with open(file_path, 'w') as file:
            file.write("Memory Profile Report\n")
            file.write("=====================\n\n")
            for item in report_data:
                for key, value in item.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")

    def _generate_csv_report(self, file_path, report_data):
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=report_data[0].keys())
            writer.writeheader()
            writer.writerows(report_data)

    def _generate_html_report(self, file_path, report_data):
        with open(file_path, 'w') as file:
            file.write("<html><head><title>Memory Profile Report</title></head><body>\n")
            file.write("<h1>Memory Profile Report</h1>\n")
            file.write("<table border='1'>\n")
            file.write("<tr>" + "".join(f"<th>{key}</th>" for key in report_data[0].keys()) + "</tr>\n")
            for item in report_data:
                file.write("<tr>" + "".join(f"<td>{value}</td>" for value in item.values()) + "</tr>\n")
            file.write("</table>\n")
            file.write("</body></html>\n")

    def _generate_pdf_report(self, file_path, report_data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Memory Profile Report", ln=True, align='C')
        pdf.ln(10)

        for item in report_data:
            for key, value in item.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
            pdf.ln(10)

        pdf.output(file_path)

    def _generate_xlsx_report(self, file_path, report_data):
        df = pd.DataFrame(report_data)
        df.to_excel(file_path, index=False)

    def set_dynamic_alerts(self, condition_function, action_function):
        """
        Sets dynamic alerts based on a user-defined condition function.
        
        Parameters:
        - condition_function: A function that takes an object and returns a boolean indicating if the alert condition is met.
        - action_function: A function to execute if the condition is met.
        
        Returns:
        - A list of objects that triggered the alert.
        """
        alerted_objects = []
        for name, obj in self.list_tracked_objects().items():
            try:
                if condition_function(obj):
                    action_function(obj)
                    alerted_objects.append({
                        'name': name,
                        'type': type(obj).__name__,
                        'id': id(obj)
                    })
            except Exception as e:
                print(f"Error processing object {name}: {e}")
        
        return alerted_objects

    def get_reference_chain(self, obj, max_depth=10):
        """
        Analyzes the chain of references leading to a given object, up to a specified depth.
        
        Parameters:
        - obj: The object to analyze.
        - max_depth: Maximum depth of the reference chain.
        
        Returns:
        - A list of lists, where each inner list represents a chain of references leading to the object.
        """
        def find_chain(obj, depth, visited):
            if depth > max_depth or id(obj) in visited:
                return []
            visited.add(id(obj))
            referrers = gc.get_referrers(obj)
            chains = []
            for referrer in referrers:
                if isinstance(referrer, dict) or isinstance(referrer, (list, tuple, set)):
                    continue  # Skip dictionaries and collections to avoid false positives
                chain = find_chain(referrer, depth + 1, visited.copy())
                for c in chain:
                    chains.append([str(id(referrer))] + c)
                if not chain:
                    chains.append([str(id(referrer))])
            return chains
        
        return find_chain(obj, 0, set())
