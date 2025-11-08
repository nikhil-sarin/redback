#!/usr/bin/env python3
"""
Pre-commit hook to detect incorrect NaN comparisons in Python files.
"""
import re
import sys


def check_file(filename):
    """Check a single Python file for incorrect NaN comparisons."""
    errors = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Pattern to detect comparisons with np.nan
        patterns = [
            (r'==\s*np\.nan\b', '== np.nan (use np.isnan() instead)'),
            (r'!=\s*np\.nan\b', '!= np.nan (use ~np.isnan() instead)'),
            (r'np\.nan\s*==', 'np.nan == (use np.isnan() instead)'),
            (r'np\.nan\s*!=', 'np.nan != (use ~np.isnan() instead)'),
        ]

        for line_num, line in enumerate(lines, start=1):
            # Skip comments
            if line.strip().startswith('#'):
                continue

            for pattern, message in patterns:
                if re.search(pattern, line):
                    errors.append(
                        f"{filename}:{line_num}: Incorrect NaN comparison: {message}\n"
                        f"  Line: {line.strip()}"
                    )

        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return False

        return True

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
        print("\n❌ Found incorrect NaN comparisons. Use np.isnan() instead.", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
