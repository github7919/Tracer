# examples/tree_visualization.py
# To run this type: python -m examples.tree_visualization


from pytrace.tree import (
    print_ast, 
    extract_function_definitions,
    extract_variable_names,
    extract_class_definitions,
    extract_imports,
    calculate_cyclomatic_complexity,
    list_function_calls,
    extract_docstrings,
    count_lines_of_code,
    extract_exception_handling
)


# Example usage
source_code = """

def check_value(x):
    if x > 10:
        return "Greater"
    else:
        return "Lesser"











"""
print_ast(source_code)