import ast
from typing import List, Dict



def extract_function_definitions(source_code: str) -> List[str]:
    """
    Extract function definitions from the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings, each representing a function definition.
    """
    tree = ast.parse(source_code)
    functions = []

    class FunctionDefExtractor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            func_def = f"def {node.name}({', '.join(arg.arg for arg in node.args.args)}):"
            functions.append(func_def)
            self.generic_visit(node)

    extractor = FunctionDefExtractor()
    extractor.visit(tree)
    return functions

def extract_variable_names(source_code: str) -> List[str]:
    """
    Extract variable names from the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings, each representing a variable name.
    """
    tree = ast.parse(source_code)
    variables = set()

    class VariableExtractor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.add(target.id)
            self.generic_visit(node)

    extractor = VariableExtractor()
    extractor.visit(tree)
    return list(variables)

def extract_class_definitions(source_code: str) -> List[str]:
    """
    Extract class definitions from the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings, each representing a class definition.
    """
    tree = ast.parse(source_code)
    classes = []

    class ClassDefExtractor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_def = f"class {node.name}:"
            classes.append(class_def)
            self.generic_visit(node)

    extractor = ClassDefExtractor()
    extractor.visit(tree)
    return classes

def extract_imports(source_code: str) -> List[str]:
    """
    Extract import statements from the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings representing import statements.
    """
    tree = ast.parse(source_code)
    imports = []

    class ImportExtractor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                imports.append(f"import {alias.name}")
            self.generic_visit(node)
        
        def visit_ImportFrom(self, node):
            for alias in node.names:
                imports.append(f"from {node.module} import {alias.name}")
            self.generic_visit(node)

    extractor = ImportExtractor()
    extractor.visit(tree)
    return imports

def calculate_cyclomatic_complexity(source_code: str) -> Dict[str, int]:
    """
    Calculate cyclomatic complexity for each function in the source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A dictionary where keys are function names and values are their cyclomatic complexities.
    """
    tree = ast.parse(source_code)
    complexities = {}

    class ComplexityCalculator(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None
            self.complexity = 0
        
        def visit_FunctionDef(self, node):
            self.current_function = node.name
            self.complexity = 1  # Start with one for the function itself
            self.generic_visit(node)
            complexities[self.current_function] = self.complexity
            self.current_function = None

        def visit_If(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_While(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_For(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_With(self, node):
            self.complexity += 1
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            self.complexity += 1
            self.generic_visit(node)

    calculator = ComplexityCalculator()
    calculator.visit(tree)
    return complexities

def list_function_calls(source_code: str) -> List[str]:
    """
    List all function calls in the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings representing function calls.
    """
    tree = ast.parse(source_code)
    calls = []

    class FunctionCallLister(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{ast.dump(node.func.value, annotate_fields=False)}.{node.func.attr}"
            else:
                func_name = str(node.func)
            calls.append(func_name)
            self.generic_visit(node)

    lister = FunctionCallLister()
    lister.visit(tree)
    return calls

def extract_docstrings(source_code: str) -> Dict[str, str]:
    """
    Extract docstrings from functions and classes in the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A dictionary where keys are function/class names and values are their docstrings.
    """
    tree = ast.parse(source_code)
    docstrings = {}

    class DocstringExtractor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if ast.get_docstring(node):
                docstrings[node.name] = ast.get_docstring(node)
            self.generic_visit(node)
        
        def visit_ClassDef(self, node):
            if ast.get_docstring(node):
                docstrings[node.name] = ast.get_docstring(node)
            self.generic_visit(node)

    extractor = DocstringExtractor()
    extractor.visit(tree)
    return docstrings

def count_lines_of_code(source_code: str) -> Dict[str, int]:
    """
    Count the number of lines of code in each function in the source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A dictionary where keys are function names and values are line counts.
    """
    tree = ast.parse(source_code)
    line_counts = {}

    class LineCounter(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            start_lineno = node.lineno
            end_lineno = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else start_lineno
            line_counts[node.name] = end_lineno - start_lineno + 1
            self.generic_visit(node)

    counter = LineCounter()
    counter.visit(tree)
    return line_counts

def extract_exception_handling(source_code: str) -> List[str]:
    """
    Extract exception handling blocks from the given source code.
    
    Parameters:
    - source_code: A string containing the source code to be analyzed.
    
    Returns:
    - A list of strings representing exception handling blocks.
    """
    tree = ast.parse(source_code)
    blocks = []

    class ExceptionHandlerExtractor(ast.NodeVisitor):
        def visit_Try(self, node):
            blocks.append("try:")
            for handler in node.handlers:
                if handler.type:
                    blocks.append(f"except {ast.dump(handler.type, annotate_fields=False)}:")
                else:
                    blocks.append("except:")
                self.generic_visit(handler)
            if node.finalbody:
                blocks.append("finally:")
                for stmt in node.finalbody:
                    blocks.append("    " + ast.dump(stmt, annotate_fields=False))
            self.generic_visit(node)

    extractor = ExceptionHandlerExtractor()
    extractor.visit(tree)
    return blocks


