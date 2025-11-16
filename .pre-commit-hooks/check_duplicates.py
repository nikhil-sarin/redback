#!/usr/bin/env python3
"""
Pre-commit hook to detect duplicate class and function definitions in Python files.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path


class DuplicateChecker(ast.NodeVisitor):
    """AST visitor to find duplicate class and function definitions."""

    def __init__(self, filename):
        self.filename = filename
        self.module_classes = defaultdict(list)
        self.module_functions = defaultdict(list)
        self.module_constants = defaultdict(list)
        self.errors = []
        self.in_class = False

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        # Only track module-level class definitions
        if not self.in_class:
            self.module_classes[node.name].append(node.lineno)

        # Don't traverse into class bodies for function tracking
        old_in_class = self.in_class
        self.in_class = True
        self.generic_visit(node)
        self.in_class = old_in_class

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        # Only track module-level functions (not methods inside classes)
        if not self.in_class:
            self.module_functions[node.name].append(node.lineno)
        # Don't visit inside functions
        pass

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        # Only track module-level async functions
        if not self.in_class:
            self.module_functions[node.name].append(node.lineno)
        # Don't visit inside functions
        pass

    def visit_Assign(self, node):
        """Visit assignments to detect duplicate constants."""
        # Only check module-level assignments (not inside functions/classes)
        if not self.in_class:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.module_constants[target.id].append(node.lineno)
        # Don't visit inside assignments
        pass

    def check_duplicates(self):
        """Check for duplicates and report errors."""
        # Check duplicate classes
        for class_name, lines in self.module_classes.items():
            if len(lines) > 1:
                self.errors.append(
                    f"{self.filename}: Duplicate class '{class_name}' "
                    f"defined at lines: {', '.join(map(str, lines))}"
                )

        # Check duplicate functions (at module level only)
        for func_name, lines in self.module_functions.items():
            if len(lines) > 1:
                self.errors.append(
                    f"{self.filename}: Duplicate module-level function '{func_name}' "
                    f"defined at lines: {', '.join(map(str, lines))}"
                )

        # Check duplicate constants (only well-known constant patterns)
        for const_name, lines in self.module_constants.items():
            if len(lines) > 1:
                # Only report if it's likely a constant (all uppercase or specific known constants)
                if const_name.isupper() or const_name in ['solar_radius', 'speed_of_light', 'planck', 'boltzmann_constant']:
                    self.errors.append(
                        f"{self.filename}: Duplicate module-level constant '{const_name}' "
                        f"assigned at lines: {', '.join(map(str, lines))}"
                    )

        return len(self.errors) == 0


def check_file(filename):
    """Check a single Python file for duplicates."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=filename)
        checker = DuplicateChecker(filename)
        checker.visit(tree)

        if not checker.check_duplicates():
            for error in checker.errors:
                print(error, file=sys.stderr)
            return False

        return True

    except SyntaxError as e:
        print(f"{filename}: Syntax error: {e}", file=sys.stderr)
        return True  # Don't fail on syntax errors - other tools will catch them
    except Exception as e:
        print(f"{filename}: Error checking file: {e}", file=sys.stderr)
        return True  # Don't fail on unexpected errors


def main():
    """Main entry point."""
    files = sys.argv[1:]
    if not files:
        print("No files to check", file=sys.stderr)
        return 0

    all_passed = True
    for filename in files:
        if not check_file(filename):
            all_passed = False

    if not all_passed:
        print("\n❌ Found duplicate definitions. Please remove duplicates.", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
