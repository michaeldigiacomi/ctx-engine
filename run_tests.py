#!/usr/bin/env python3
"""
Test runner for pgvector-context-engine.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests (requires PostgreSQL)
    python run_tests.py --coverage   # Run with coverage report
"""

import argparse
import subprocess
import sys


def run_tests(args):
    """Run tests with pytest."""
    cmd = ["python", "-m", "pytest"]

    if args.unit:
        cmd.extend(["-m", "not integration"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    if args.coverage:
        cmd.extend(["--cov=context_engine", "--cov-report=term-missing"])

    if args.verbose:
        cmd.append("-v")

    if args.tests:
        cmd.extend(args.tests)
    else:
        cmd.append("tests/")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd)
    return result.returncode


def check_dependencies():
    """Check that required dependencies are installed."""
    try:
        import psycopg2
        print("psycopg2: installed")
    except ImportError:
        print("psycopg2: NOT installed (run: pip install psycopg2-binary)")
        return False

    try:
        import pytest
        print("pytest: installed")
    except ImportError:
        print("pytest: NOT installed (run: pip install pytest)")
        return False

    if args.coverage:
        try:
            import pytest_cov
            print("pytest-cov: installed")
        except ImportError:
            print("pytest-cov: NOT installed (run: pip install pytest-cov)")
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for pgvector-context-engine")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("tests", nargs="*", help="Specific test files or directories")

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    print("-" * 60)
    sys.exit(run_tests(args))
