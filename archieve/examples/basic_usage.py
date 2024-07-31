from pytrace.memory import print_memory_usage, compare_memory_usage, largest_object, object_size_summary

if __name__ == "__main__":
    # Example usage
    example_list = [1, 2, 3, [4, 5]]
    print_memory_usage(example_list)
    
    another_list = [6, 7, 8]
    print("Comparison of memory usage:")
    print(compare_memory_usage(example_list, another_list))
    
    objects = [example_list, another_list, "string", {"key": "value"}]
    print("Largest object:")
    print(largest_object(objects))
    
    print("Object size summary:")
    print(object_size_summary(objects))
