# pytrace/memory.py
import gc
import logging
import os
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Container, Mapping

import matplotlib.pyplot as plt
import numpy as np
import psutil
from pympler import asizeof, summary


"""
List of functions in this module and their purposes:

1. memory_usage(obj, seen=None): Calculates the memory usage of an object and its contents recursively.
2. detailed_memory_usage(obj): Calculates the detailed memory usage of an object, including referenced objects.
3. print_memory_usage(obj, include_detailed=True): Prints the memory usage of an object.
4. compare_memory_usage(obj1, obj2, labels=None): Compares the memory usage of two objects.
5. largest_object(objects): Finds the largest object in a collection based on memory usage.
6. analyze_collection(objects, labels=None): Analyzes the memory usage of a collection of objects.
7. object_size_summary(objects): Gets a summary of memory usage for different types of objects.
8. track_memory_usage_over_time(func, *args, interval=0.1, duration=None, top_n=10, track_system_memory=True, plot_results=False, **kwargs): Tracks detailed memory usage of a function over time.
9. plot_memory_usage(memory_data, execution_time): Plots memory usage data over time.
10. detect_memory_leaks(threshold=10, object_types=(list, dict, set, tuple), size_threshold=1024*1024, top_n=10, track_locations=True): Detects potential memory leaks with detailed analysis.
11. heap_summary(detailed=False, top_n=10, size_threshold=1024): Provides a comprehensive summary of the current heap state.
12. print_heap_summary(summary): Prints a formatted heap summary.
13. continuous_memory_monitor(log_interval=60, duration=None, log_file="memory_log.txt", track_system_memory=True): Continuously monitors memory usage over long periods.
14. setup_memory_logging(log_interval=3600, log_history=24, log_level=logging.INFO): Sets up automated memory logging.
15. visualize_memory_usage(data, plot_type='bar', title='Memory Usage Visualization', save_path=None, show_plot=True): Creates custom visualizations for memory usage data.
"""

def memory_usage(obj, seen=None):
    """
    Calculate the memory usage of an object and its contents recursively.
    
    Parameters:
    - obj: The object whose memory usage is to be calculated.
    - seen: A set of object ids to keep track of objects already counted (used internally for recursion).
    
    Returns:
    - The memory size of the object and its contents in bytes.
    """
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, (str, bytes, int, float, bool)):
        pass  # These types don't have any nested objects
    elif isinstance(obj, Mapping):
        size += sum(memory_usage(key, seen) + memory_usage(value, seen) for key, value in obj.items())
    elif isinstance(obj, Container) and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(memory_usage(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += memory_usage(vars(obj), seen)
    elif hasattr(obj, '__slots__'):
        size += sum(memory_usage(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))
    
    return size

def detailed_memory_usage(obj):
    """
    Calculate the detailed memory usage of an object, including the size
    of referenced objects.
    
    Parameters:
    - obj: The object whose detailed memory usage is to be calculated.
    
    Returns:
    - The total memory size of the object and all referenced objects in bytes.
    """
    return asizeof.asizeof(obj)

def print_memory_usage(obj, include_detailed=True):
    """
    Print the memory usage of an object.
    
    Parameters:
    - obj: The object whose memory usage is to be printed.
    - include_detailed: Boolean flag to include detailed memory usage (default: True).
    """
    size = memory_usage(obj)
    print(f"Memory usage: {size:,} bytes")
    
    if include_detailed:
        detailed_size = detailed_memory_usage(obj)
        print(f"Detailed memory usage: {detailed_size:,} bytes")



def compare_memory_usage(obj1, obj2, labels=None):
    """
    Compare the memory usage of two objects.
    
    Parameters:
    - obj1: The first object to compare.
    - obj2: The second object to compare.
    - labels: Optional tuple of labels for the objects (default: None).
    """
    if labels is None:
        labels = ("Object 1", "Object 2")
    
    size1 = memory_usage(obj1)
    size2 = memory_usage(obj2)
    detailed_size1 = detailed_memory_usage(obj1)
    detailed_size2 = detailed_memory_usage(obj2)
    
    print(f"{labels[0]} memory usage: {size1:,} bytes")
    print(f"{labels[1]} memory usage: {size2:,} bytes")
    print(f"{labels[0]} detailed memory usage: {detailed_size1:,} bytes")
    print(f"{labels[1]} detailed memory usage: {detailed_size2:,} bytes")
    
    diff = size2 - size1
    detailed_diff = detailed_size2 - detailed_size1
    
    print(f"\nDifference ({labels[1]} - {labels[0]}):")
    print(f"Memory usage difference: {diff:,} bytes")
    print(f"Detailed memory usage difference: {detailed_diff:,} bytes")



def largest_object(objects):
    """
    Find the largest object in a collection of objects based on memory usage.
    
    Parameters:
    - objects: A list of objects to be compared.
    
    Returns:
    - A tuple containing the largest object and its size in bytes.
    """
    largest = None
    max_size = 0
    for obj in objects:
        size = detailed_memory_usage(obj)
        if size > max_size:
            max_size = size
            largest = obj
    return largest, max_size




def analyze_collection(objects, labels=None):
    """
    Analyze the memory usage of a collection of objects.
    
    Parameters:
    - objects: A list of objects to be analyzed.
    - labels: Optional list of labels for the objects (default: None).
    """
    if labels is None:
        labels = [f"Object {i+1}" for i in range(len(objects))]
    
    print("Memory Usage Analysis:")
    for obj, label in zip(objects, labels):
        size = detailed_memory_usage(obj)
        print(f"{label}: {size:,} bytes")
    
    largest, size = largest_object(objects)
    largest_index = objects.index(largest)
    print(f"\nLargest object: {labels[largest_index]} ({size:,} bytes)")

    total_size = sum(detailed_memory_usage(obj) for obj in objects)
    print(f"Total memory usage of all objects: {total_size:,} bytes")




def object_size_summary(objects):
    """
    Get a summary of memory usage for different types of objects.
    
    Parameters:
    - objects: A list of objects to be summarized.
    
    Returns:
    - A dictionary with object types as keys and their total memory usage as values.
    """
    summary_data = defaultdict(int)
    
    for obj in objects:
        try:
            obj_type = type(obj).__name__
            size = asizeof.asizeof(obj)
            summary_data[obj_type] += size
        except Exception as e:
            print(f"Error processing object of type {type(obj).__name__}: {e}")
    
    return dict(summary_data)





def track_memory_usage_over_time(func, *args, interval=0.1, duration=None, top_n=10, 
                                 track_system_memory=True, plot_results=False, **kwargs):
    """
    Track detailed memory usage of a function over time.
    
    Parameters:
    - func: The function to be measured.
    - *args: Positional arguments to pass to the function.
    - interval: Time interval (in seconds) between memory measurements (default: 0.1).
    - duration: Maximum duration (in seconds) to track memory (default: None, runs until function completes).
    - top_n: Number of top memory consumers to display (default: 10).
    - track_system_memory: Whether to track overall system memory usage (default: True).
    - plot_results: Whether to plot memory usage over time (default: False).
    - **kwargs: Keyword arguments to pass to the function.
    
    Returns:
    - A tuple containing:
        1. The result of the function call.
        2. A dictionary of memory usage data over time.
        3. The total execution time.
    """
    gc.collect()  # Garbage collect before starting
    tracemalloc.start()
    start_time = time.time()
    start_snapshot = tracemalloc.take_snapshot()
    
    memory_data = defaultdict(list)
    stop_flag = threading.Event()

    def measure_memory():
        while not stop_flag.is_set():
            current_time = time.time() - start_time
            current = tracemalloc.take_snapshot()
            stats = current.compare_to(start_snapshot, 'lineno')
            memory_data['tracemalloc'].append((current_time, stats))
            
            if track_system_memory:
                process = psutil.Process()
                memory_data['process_memory'].append((current_time, process.memory_info().rss))
                memory_data['system_memory'].append((current_time, psutil.virtual_memory().percent))
            
            time.sleep(interval)

    memory_thread = threading.Thread(target=measure_memory)
    memory_thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_flag.set()
        memory_thread.join()
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

    execution_time = time.time() - start_time

    print(f"\nMemory usage summary (top {top_n} consumers):")
    top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    for stat in top_stats[:top_n]:
        print(stat)

    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    if track_system_memory:
        print("\nSystem Memory Usage:")
        print(f"Peak Process Memory: {max(m for _, m in memory_data['process_memory']) / (1024 * 1024):.2f} MB")
        print(f"Peak System Memory: {max(m for _, m in memory_data['system_memory']):.2f}%")

    if plot_results:
        plot_memory_usage(memory_data, execution_time)

    return result, dict(memory_data), execution_time




def plot_memory_usage(memory_data, execution_time):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    if 'process_memory' in memory_data:
        times, memory = zip(*memory_data['process_memory'])
        plt.plot(times, [m / (1024 * 1024) for m in memory], label='Process Memory (MB)')
    
    if 'system_memory' in memory_data:
        times, memory = zip(*memory_data['system_memory'])
        plt.plot(times, memory, label='System Memory (%)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage')
    plt.title(f'Memory Usage Over Time (Total Execution: {execution_time:.2f}s)')
    plt.legend()
    plt.grid(True)
    plt.show()




def detect_memory_leaks(threshold=10, object_types=(list, dict, set, tuple), 
                        size_threshold=1024*1024, top_n=10, track_locations=True):
    """
    Advanced memory leak detection with detailed analysis.
    
    Parameters:
    - threshold: The minimum number of references an object must have to be considered for leak detection.
    - object_types: A tuple of object types to check for leaks.
    - size_threshold: Minimum size (in bytes) for an object to be considered a significant leak.
    - top_n: Number of top leaks to report in detail.
    - track_locations: Whether to track and report object creation locations.
    
    Returns:
    - A tuple containing:
        1. A list of potential memory leaks.
        2. A summary of leak categories.
        3. Total memory usage of detected leaks.
    """
    gc.collect()  # Force a garbage collection cycle
    
    if track_locations:
        tracemalloc.start()
    
    leaks = []
    leak_summary = defaultdict(lambda: {'count': 0, 'total_size': 0})
    total_leak_size = 0
    
    for obj in gc.get_objects():
        try:
            if isinstance(obj, object_types):
                ref_count = sys.getrefcount(obj)
                if ref_count > threshold:
                    obj_size = asizeof.asizeof(obj)
                    if obj_size >= size_threshold:
                        obj_type = type(obj).__name__
                        leak_info = {
                            'object': obj,
                            'type': obj_type,
                            'ref_count': ref_count,
                            'size': obj_size,
                            'location': None
                        }
                        
                        if track_locations:
                            snapshot = tracemalloc.take_snapshot()
                            for stat in snapshot.statistics('lineno'):
                                if stat.traceback[-1].filename != __file__:
                                    leak_info['location'] = f"{stat.traceback[-1].filename}:{stat.traceback[-1].lineno}"
                                    break
                        
                        leaks.append(leak_info)
                        leak_summary[obj_type]['count'] += 1
                        leak_summary[obj_type]['total_size'] += obj_size
                        total_leak_size += obj_size
        except Exception as e:
            print(f"Error processing object: {e}")
    
    if track_locations:
        tracemalloc.stop()
    
    # Sort leaks by size
    leaks.sort(key=lambda x: x['size'], reverse=True)
    
    # Prepare detailed report
    detailed_report = []
    for i, leak in enumerate(leaks[:top_n], 1):
        report = f"Leak {i}:\n"
        report += f"  Type: {leak['type']}\n"
        report += f"  Size: {leak['size']:,} bytes\n"
        report += f"  Reference Count: {leak['ref_count']}\n"
        if leak['location']:
            report += f"  Location: {leak['location']}\n"
        report += f"  Content Preview: {str(leak['object'])[:100]}...\n"
        detailed_report.append(report)
    
    # Prepare summary report
    summary_report = "Leak Summary:\n"
    for obj_type, info in leak_summary.items():
        summary_report += f"  {obj_type}: Count: {info['count']}, Total Size: {info['total_size']:,} bytes\n"
    summary_report += f"\nTotal Leak Size: {total_leak_size:,} bytes"
    
    return leaks, summary_report, detailed_report, total_leak_size


    

def heap_summary(detailed=False, top_n=10, size_threshold=1024):
    """
    Provide a comprehensive summary of the current heap state.
    
    Parameters:
    - detailed: If True, provide additional details about top memory consumers (default: False)
    - top_n: Number of top memory consumers to report in detail (default: 10)
    - size_threshold: Minimum size in bytes for an object to be included in the summary (default: 1024)
    
    Returns:
    - A dictionary containing heap summary information
    """
    gc.collect()  # Force a garbage collection cycle
    
    heap_info = defaultdict(lambda: {'count': 0, 'total_size': 0})
    large_objects = []
    
    for obj in gc.get_objects():
        try:
            obj_type = type(obj).__name__
            obj_size = asizeof.asizeof(obj)
            
            if obj_size >= size_threshold:
                heap_info[obj_type]['count'] += 1
                heap_info[obj_type]['total_size'] += obj_size
                
                if detailed:
                    large_objects.append((obj_type, obj_size, obj))
        except Exception as e:
            print(f"Error processing object: {e}")
    
    # Sort heap_info by total size
    sorted_heap_info = dict(sorted(heap_info.items(), key=lambda x: x[1]['total_size'], reverse=True))
    
    # Prepare summary
    summary = {
        'total_objects': sum(info['count'] for info in heap_info.values()),
        'total_size': sum(info['total_size'] for info in heap_info.values()),
        'type_summary': sorted_heap_info,
        'process_memory': psutil.Process().memory_info().rss,
        'virtual_memory': dict(psutil.virtual_memory()._asdict())
    }
    
    if detailed:
        # Sort and get top N large objects
        large_objects.sort(key=lambda x: x[1], reverse=True)
        summary['top_objects'] = [
            {
                'type': obj_type,
                'size': obj_size,
                'repr': repr(obj)[:100]  # Truncate representation to 100 characters
            }
            for obj_type, obj_size, obj in large_objects[:top_n]
        ]
    
    return summary



def print_heap_summary(summary):
    """
    Print a formatted heap summary.
    
    Parameters:
    - summary: The heap summary dictionary returned by heap_summary()
    """
    print("Heap Summary:")
    print(f"Total Objects: {summary['total_objects']:,}")
    print(f"Total Heap Size: {summary['total_size']:,} bytes ({summary['total_size'] / (1024 * 1024):.2f} MB)")
    print(f"Process Memory: {summary['process_memory']:,} bytes ({summary['process_memory'] / (1024 * 1024):.2f} MB)")
    print(f"System Memory Usage: {summary['virtual_memory']['percent']}%")
    
    print("\nTop 10 Object Types by Memory Usage:")
    for i, (obj_type, info) in enumerate(list(summary['type_summary'].items())[:10], 1):
        print(f"{i}. {obj_type}:")
        print(f"   Count: {info['count']:,}")
        print(f"   Total Size: {info['total_size']:,} bytes ({info['total_size'] / (1024 * 1024):.2f} MB)")
    
    if 'top_objects' in summary:
        print("\nTop Individual Objects by Size:")
        for i, obj in enumerate(summary['top_objects'], 1):
            print(f"{i}. Type: {obj['type']}")
            print(f"   Size: {obj['size']:,} bytes ({obj['size'] / (1024 * 1024):.2f} MB)")
            print(f"   Preview: {obj['repr']}")




def continuous_memory_monitor(log_interval=60, duration=None, log_file="memory_log.txt", track_system_memory=True):
    """
    Continuously monitor memory usage over long periods.
    
    Parameters:
    - log_interval: Time interval (in seconds) between memory measurements (default: 60).
    - duration: Maximum duration (in seconds) to monitor memory (default: None, runs indefinitely).
    - log_file: File to log memory usage data (default: "memory_log.txt").
    - track_system_memory: Whether to track overall system memory usage (default: True).
    
    Returns:
    - None
    """
    gc.collect()  # Garbage collect before starting
    tracemalloc.start()
    start_time = time.time()
    
    memory_data = defaultdict(list)
    stop_flag = threading.Event()

    def measure_memory():
        with open(log_file, "a") as log:
            while not stop_flag.is_set():
                current_time = time.time() - start_time
                current = tracemalloc.take_snapshot()
                stats = current.statistics('lineno')
                memory_data['tracemalloc'].append((current_time, stats))
                
                if track_system_memory:
                    process = psutil.Process()
                    memory_data['process_memory'].append((current_time, process.memory_info().rss))
                    memory_data['system_memory'].append((current_time, psutil.virtual_memory().percent))
                
                log.write(f"Time: {current_time:.2f} seconds\n")
                log.write("Top memory consumers:\n")
                for stat in stats[:10]:
                    log.write(f"{stat}\n")
                
                if track_system_memory:
                    log.write(f"Process Memory: {process.memory_info().rss / (1024 * 1024):.2f} MB\n")
                    log.write(f"System Memory: {psutil.virtual_memory().percent:.2f}%\n")
                
                log.write("\n")
                log.flush()
                time.sleep(log_interval)
    
    memory_thread = threading.Thread(target=measure_memory)
    memory_thread.start()

    try:
        if duration:
            time.sleep(duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        memory_thread.join()
        tracemalloc.stop()




def setup_memory_logging(log_interval=3600, log_history=24, log_level=logging.INFO):
    """
    Set up automated memory logging.

    Parameters:
    - log_interval: Time in seconds between log entries (default: 3600, i.e., 1 hour)
    - log_history: Number of log entries to keep in memory (default: 24)
    - log_level: Logging level for memory reports (default: logging.INFO)

    Returns:
    - None
    """
    logger = logging.getLogger('memory_logger')
    logger.setLevel(log_level)

    # Create a file handler
    file_handler = logging.FileHandler('memory_report.log')
    file_handler.setLevel(log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initialize a deque to store historical data
    history = deque(maxlen=log_history)

    def log_memory():
        while True:
            heap_info = heap_summary()
            history.append(heap_info)

            logger.info("Memory Usage Report:")
            logger.info(f"Total Objects: {heap_info['total_objects']:,}")
            logger.info(f"Total Heap Size: {heap_info['total_size'] / (1024 * 1024):.2f} MB")
            logger.info(f"Process Memory: {heap_info['process_memory'] / (1024 * 1024):.2f} MB")
            logger.info(f"System Memory Usage: {heap_info['virtual_memory']['percent']}%")

            # Log top 5 object types by memory usage
            logger.info("Top 5 Object Types by Memory Usage:")
            for i, (obj_type, info) in enumerate(list(heap_info['type_summary'].items())[:5], 1):
                logger.info(f"{i}. {obj_type}: Count: {info['count']:,}, "
                            f"Size: {info['total_size'] / (1024 * 1024):.2f} MB")

            # Calculate and log memory growth
            if len(history) > 1:
                prev_total = history[-2]['total_size']
                current_total = heap_info['total_size']
                growth = (current_total - prev_total) / (1024 * 1024)  # MB
                growth_percent = (current_total - prev_total) / prev_total * 100
                logger.info(f"Memory Growth: {growth:.2f} MB ({growth_percent:.2f}%)")

            time.sleep(log_interval)

    # Start the logging thread
    logging_thread = threading.Thread(target=log_memory, daemon=True)
    logging_thread.start()


def visualize_memory_usage(data, plot_type='bar', title='Memory Usage Visualization', 
                           save_path=None, show_plot=True):
    """
    Create custom visualizations for memory usage data.

    Parameters:
    - data: Dictionary containing memory usage data. The structure depends on the plot_type.
    - plot_type: Type of plot to generate. Options: 'bar', 'pie', 'timeline', 'heatmap'
    - title: Title of the plot
    - save_path: Path to save the plot image. If None, the plot is not saved.
    - show_plot: Boolean to determine if the plot should be displayed.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.title(title)

    if plot_type == 'bar':
        # Bar chart for comparing memory usage of different objects/types
        objects = list(data.keys())
        usage = [data[obj] for obj in objects]
        plt.bar(objects, usage)
        plt.xlabel('Objects/Types')
        plt.ylabel('Memory Usage (bytes)')
        plt.xticks(rotation=45, ha='right')

    elif plot_type == 'pie':
        # Pie chart for showing proportion of memory usage
        objects = list(data.keys())
        usage = [data[obj] for obj in objects]
        plt.pie(usage, labels=objects, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')

    elif plot_type == 'timeline':
        # Line plot for showing memory usage over time
        times = [t for t, _ in data]
        usage = [u for _, u in data]
        plt.plot(times, usage)
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (bytes)')

    elif plot_type == 'heatmap':
        # Heatmap for visualizing memory usage patterns
        plt.imshow(data, aspect='auto', cmap='YlOrRd')
        plt.colorbar(label='Memory Usage (bytes)')
        plt.xlabel('Time')
        plt.ylabel('Object/Type')

    else:
        raise ValueError("Invalid plot_type. Choose from 'bar', 'pie', 'timeline', or 'heatmap'.")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

