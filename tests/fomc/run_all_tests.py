#!/usr/bin/env python3
"""
Master test runner for FOMC comprehensive test suite.
Executes all test levels with proper reporting.
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class TestRunner:
    """Orchestrates test execution and reporting."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.test_dir = Path(__file__).parent

    def run_test_suite(
        self, suite_name: str, test_path: str, extra_args: List[str] = None
    ) -> Dict[str, Any]:
        """Run a specific test suite."""
        print(f"\n{'=' * 60}")
        print(f"Running {suite_name} Tests")
        print(f"{'=' * 60}")

        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]

        if extra_args:
            cmd.extend(extra_args)

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.test_dir.parent.parent,  # Project root
        )
        elapsed = time.time() - start

        # Parse output for test counts
        passed = failed = skipped = 0
        for line in result.stdout.split("\n"):
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part and i > 0:
                        try:
                            passed = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass
            if "failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "failed" in part and i > 0:
                        try:
                            failed = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass
            if "skipped" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "skipped" in part and i > 0:
                        try:
                            skipped = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass

        suite_result = {
            "suite": suite_name,
            "path": test_path,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped,
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "return_code": result.returncode,
        }

        if self.verbose:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        # Summary
        print(f"\n{suite_name} Results:")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {'✅ PASSED' if suite_result['success'] else '❌ FAILED'}")

        return suite_result

    def run_all_tests(self, skip_live: bool = True):
        """Run all test suites in order."""
        self.start_time = time.time()

        print("\n" + "🚀 " * 20)
        print("FOMC COMPREHENSIVE TEST SUITE")
        print("🚀 " * 20)

        # Define test suites in order of execution
        test_suites = [
            {
                "name": "Smoke",
                "path": "tests/fomc/smoke/",
                "args": ["--tb=short"],
                "critical": True,
            },
            {
                "name": "Unit",
                "path": "tests/fomc/unit/",
                "args": ["--cov=flame.code.fomc", "--cov-report=term-missing"],
                "critical": True,
            },
            {
                "name": "Integration",
                "path": "tests/fomc/integration/",
                "args": [],
                "critical": True,
            },
            {
                "name": "Performance",
                "path": "tests/fomc/performance/",
                "args": [],
                "critical": False,
            },
        ]

        # Add E2E tests if not skipping live tests
        if not skip_live:
            test_suites.append(
                {
                    "name": "E2E",
                    "path": "tests/fomc/e2e/",
                    "args": [],
                    "critical": False,
                }
            )

        # Run each suite
        all_passed = True
        for suite in test_suites:
            result = self.run_test_suite(
                suite["name"], suite["path"], suite.get("args", [])
            )

            self.results[suite["name"].lower()] = result

            if not result["success"] and suite.get("critical", False):
                all_passed = False
                print(f"\n⚠️ Critical test suite '{suite['name']}' failed!")
                if not self.verbose:
                    print("Run with --verbose to see detailed output")

        # Generate summary
        self.generate_summary(all_passed)

        return all_passed

    def generate_summary(self, all_passed: bool):
        """Generate and display test summary."""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_passed = sum(r["passed"] for r in self.results.values())
        total_failed = sum(r["failed"] for r in self.results.values())
        total_skipped = sum(r["skipped"] for r in self.results.values())
        total_tests = total_passed + total_failed + total_skipped

        print(f"\nTotal Tests: {total_tests}")
        print(f"  ✅ Passed: {total_passed}")
        print(f"  ❌ Failed: {total_failed}")
        print(f"  ⏭️ Skipped: {total_skipped}")
        print(f"\nTotal Time: {total_time:.2f}s")

        print("\nSuite Breakdown:")
        for name, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            print(
                f"  {status} {name.capitalize()}: "
                f"{result['passed']}/{result['total']} passed "
                f"({result['elapsed']:.2f}s)"
            )

        if all_passed:
            print("\n🎉 ALL TEST SUITES PASSED! 🎉")
            print("The FOMC implementation is ready for Phase 2 migration.")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("Please review and fix the failing tests before proceeding.")

        # Save report
        self.save_report(all_passed, total_time)

    def save_report(self, all_passed: bool, total_time: float):
        """Save test report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "success": all_passed,
            "total_time": total_time,
            "suites": self.results,
            "summary": {
                "total_passed": sum(r["passed"] for r in self.results.values()),
                "total_failed": sum(r["failed"] for r in self.results.values()),
                "total_skipped": sum(r["skipped"] for r in self.results.values()),
            },
        }

        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"fomc_test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Test report saved to: {report_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive FOMC test suite")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed test output"
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Include E2E tests with live API calls",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "unit", "integration", "performance", "e2e", "all"],
        default="all",
        help="Run specific test suite",
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    if args.suite == "all":
        success = runner.run_all_tests(skip_live=not args.include_live)
    else:
        # Run single suite
        suite_map = {
            "smoke": ("Smoke", "tests/fomc/smoke/"),
            "unit": ("Unit", "tests/fomc/unit/"),
            "integration": ("Integration", "tests/fomc/integration/"),
            "performance": ("Performance", "tests/fomc/performance/"),
            "e2e": ("E2E", "tests/fomc/e2e/"),
        }

        if args.suite in suite_map:
            name, path = suite_map[args.suite]
            result = runner.run_test_suite(name, path)
            success = result["success"]

            runner.results = {args.suite: result}
            runner.generate_summary(success)
        else:
            print(f"Unknown suite: {args.suite}")
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
