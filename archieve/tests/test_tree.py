import ast

def print_ast(source_code: str):
    def pretty_print(node, indent="", last=True, vars_in_scope=None):
        if vars_in_scope is None:
            vars_in_scope = set()

        def is_last_child(index, iterable):
            return index == len(iterable) - 1

        prefix = "└─" if last else "├─"
        next_indent = indent + ("  " if last else "│ ")

        if isinstance(node, ast.FunctionDef):
            print(f"{indent}{prefix}function({', '.join(arg.arg for arg in node.args.args)})")
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
        elif isinstance(node, ast.ClassDef):
            bases = ', '.join([b.id for b in node.bases if isinstance(b, ast.Name)])
            print(f"{indent}{prefix}class {node.name}({bases})")
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
        elif isinstance(node, ast.Return):
            print(f"{indent}{prefix}return")
            if node.value:
                pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.If):
            print(f"{indent}{prefix}`if`")
            pretty_print(node.test, next_indent, last=False, vars_in_scope=vars_in_scope)
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
            if node.orelse:
                print(f"{next_indent}└─█─`{{`")
                for i, stmt in enumerate(node.orelse):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, node.orelse), vars_in_scope)
        elif isinstance(node, ast.Compare):
            op_str = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
                      ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in', ast.NotIn: 'not in'}.get(type(node.ops[0]), type(node.ops[0]).__name__)
            print(f"{indent}{prefix}`{op_str}`")
            pretty_print(node.left, next_indent, last=False, vars_in_scope=vars_in_scope)
            for i, comp in enumerate(node.comparators):
                pretty_print(comp, next_indent, last=is_last_child(i, node.comparators), vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.BinOp):
            op_str = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Mod: '%%',
                      ast.Pow: '^', ast.FloorDiv: '//', ast.BitOr: '|', ast.BitAnd: '&', ast.BitXor: '^',
                      ast.LShift: '<<', ast.RShift: '>>'}.get(type(node.op), type(node.op).__name__)
            print(f"{indent}{prefix}`{op_str}`")
            pretty_print(node.left, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.right, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.UnaryOp):
            op_str = {ast.UAdd: '+', ast.USub: '-', ast.Not: '!', ast.Invert: '~'}.get(type(node.op), type(node.op).__name__)
            print(f"{indent}{prefix}`{op_str}`")
            pretty_print(node.operand, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else str(node.func)
            print(f"{indent}{prefix}{func_name}")
            for i, arg in enumerate(node.args):
                pretty_print(arg, next_indent, last=is_last_child(i, node.args) and not node.keywords, vars_in_scope=vars_in_scope)
            for i, kw in enumerate(node.keywords):
                print(f"{next_indent}{'└─' if is_last_child(i, node.keywords) else '├─'}{kw.arg}=")
                pretty_print(kw.value, next_indent + "  ", last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Name):
            print(f"{indent}{prefix}{node.id}")
        elif isinstance(node, ast.Attribute):
            print(f"{indent}{prefix}{node.attr}")
            pretty_print(node.value, next_indent, last=False, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Subscript):
            print(f"{indent}{prefix}Subscript")
            pretty_print(node.value, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.slice, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Constant):
            print(f"{indent}{prefix}{repr(node.value)}")
        elif isinstance(node, ast.List):
            print(f"{indent}{prefix}list")
            for i, elt in enumerate(node.elts):
                pretty_print(elt, next_indent, last=is_last_child(i, node.elts), vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Tuple):
            print(f"{indent}{prefix}tuple")
            for i, elt in enumerate(node.elts):
                pretty_print(elt, next_indent, last=is_last_child(i, node.elts), vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Set):
            print(f"{indent}{prefix}set")
            for i, elt in enumerate(node.elts):
                pretty_print(elt, next_indent, last=is_last_child(i, node.elts), vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Dict):
            print(f"{indent}{prefix}dict")
            for i, (key, value) in enumerate(zip(node.keys, node.values)):
                print(f"{next_indent}{'└─' if is_last_child(i, node.keys) else '├─'}█─`:`")
                pretty_print(key, next_indent + "  ", last=False, vars_in_scope=vars_in_scope)
                pretty_print(value, next_indent + "  ", last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            print(f"{indent}{prefix}{type(node).__name__}")
            pretty_print(node.elt, next_indent, last=False, vars_in_scope=vars_in_scope)
            for gen in node.generators:
                pretty_print(gen, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.DictComp):
            print(f"{indent}{prefix}DictComp")
            print(f"{next_indent}├─key:")
            pretty_print(node.key, next_indent + "  ", last=False, vars_in_scope=vars_in_scope)
            print(f"{next_indent}├─value:")
            pretty_print(node.value, next_indent + "  ", last=False, vars_in_scope=vars_in_scope)
            for gen in node.generators:
                pretty_print(gen, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.comprehension):
            print(f"{indent}{prefix}for {ast.unparse(node.target)} in")
            pretty_print(node.iter, next_indent, last=False, vars_in_scope=vars_in_scope)
            for if_clause in node.ifs:
                print(f"{next_indent}├─if")
                pretty_print(if_clause, next_indent + "  ", last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.For):
            print(f"{indent}{prefix}`for`")
            pretty_print(node.target, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.iter, next_indent, last=False, vars_in_scope=vars_in_scope)
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
            if node.orelse:
                print(f"{next_indent}└─█─`else`")
                for i, stmt in enumerate(node.orelse):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, node.orelse), vars_in_scope)
        elif isinstance(node, ast.While):
            print(f"{indent}{prefix}`while`")
            pretty_print(node.test, next_indent, last=False, vars_in_scope=vars_in_scope)
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
            if node.orelse:
                print(f"{next_indent}└─█─`else`")
                for i, stmt in enumerate(node.orelse):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, node.orelse), vars_in_scope)
        elif isinstance(node, ast.Assign):
            print(f"{indent}{prefix}`<-`")
            for target in node.targets:
                pretty_print(target, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.AugAssign):
            op_str = {ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/=', ast.Mod: '%=',
                      ast.Pow: '**=', ast.FloorDiv: '//='}.get(type(node.op), type(node.op).__name__ + '=')
            print(f"{indent}{prefix}`{op_str}`")
            pretty_print(node.target, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Try):
            print(f"{indent}{prefix}try")
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
            for handler in node.handlers:
                print(f"{next_indent}└─except {handler.type.id if handler.type else ''}")
                for i, stmt in enumerate(handler.body):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, handler.body), vars_in_scope)
            if node.orelse:
                print(f"{next_indent}└─else")
                for i, stmt in enumerate(node.orelse):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, node.orelse), vars_in_scope)
            if node.finalbody:
                print(f"{next_indent}└─finally")
                for i, stmt in enumerate(node.finalbody):
                    pretty_print(stmt, next_indent + "  ", is_last_child(i, node.finalbody), vars_in_scope)
        elif isinstance(node, ast.With):
            print(f"{indent}{prefix}with")
            for item in node.items:
                pretty_print(item.context_expr, next_indent, last=False, vars_in_scope=vars_in_scope)
                if item.optional_vars:
                    print(f"{next_indent}├─as")
                    pretty_print(item.optional_vars, next_indent + "  ", last=False, vars_in_scope=vars_in_scope)
            print(f"{next_indent}└─█─`{{`")
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, next_indent + "  ", is_last_child(i, node.body), vars_in_scope)
        elif isinstance(node, ast.Lambda):
            print(f"{indent}{prefix}lambda")
            for i, arg in enumerate(node.args.args):
                print(f"{next_indent}{'└─' if is_last_child(i, node.args.args) else '├─'}{arg.arg}")
            pretty_print(node.body, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Pass):
            print(f"{indent}{prefix}pass")
        elif isinstance(node, ast.Break):
            print(f"{indent}{prefix}break")
        elif isinstance(node, ast.Continue):
            print(f"{indent}{prefix}continue")
        elif isinstance(node, ast.Raise):
            print(f"{indent}{prefix}raise")
            if node.exc:
                pretty_print(node.exc, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Yield):
            print(f"{indent}{prefix}yield")
            if node.value:
                pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.YieldFrom):
            print(f"{indent}{prefix}yield from")
            pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Global):
            print(f"{indent}{prefix}global {', '.join(node.names)}")
        elif isinstance(node, ast.Nonlocal):
            print(f"{indent}{prefix}nonlocal {', '.join(node.names)}")
        elif isinstance(node, ast.Assert):
            print(f"{indent}{prefix}assert")
            pretty_print(node.test, next_indent, last=False, vars_in_scope=vars_in_scope)
            if node.msg:
                pretty_print(node.msg, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.IfExp):
            print(f"{indent}{prefix}ifexp")
            pretty_print(node.test, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.body, next_indent, last=False, vars_in_scope=vars_in_scope)
            pretty_print(node.orelse, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.JoinedStr):
            print(f"{indent}{prefix}f-string")
            for i, value in enumerate(node.values):
                pretty_print(value, next_indent, last=is_last_child(i, node.values), vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.FormattedValue):
            print(f"{indent}{prefix}FormattedValue")
            pretty_print(node.value, next_indent, last=True, vars_in_scope=vars_in_scope)
        elif isinstance(node, ast.Module):
            for i, stmt in enumerate(node.body):
                pretty_print(stmt, indent, is_last_child(i, node.body), vars_in_scope)
        else:
            print(f"{indent}{prefix}Unsupported node type: {type(node).__name__}")

    try:
        tree = ast.parse(source_code)
        pretty_print(tree)
    except SyntaxError as e:
        print(f"Syntax error in the provided code: {e}")

# Test each example
print("Example 1:")
print_ast("""
def greet(name):
    return f"Hello, {name}!"
""")
print("\nExample 2:")
print_ast("""
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
""")
print("\nExample 3:")
print_ast("""
numbers = [1, 2, 3, 4, 5]
squared = [(lambda x: x**2)(num) for num in numbers if num % 2 == 0]
""")
print("\nExample 4:")
print_ast("""
try:
    with open('example.txt', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("File not found")
finally:
    print("Operation completed")
""")
print("\nExample 5:")
print_ast("""
def process_data(data, threshold=10):
    result = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            if value > threshold:
                result[key] = value
        elif isinstance(value, list):
            result[key] = [x for x in value if x > threshold]
    
    def summarize(d):
        return {k: sum(v) if isinstance(v, list) else v for k, v in d.items()}
    
    return summarize(result)
""")
