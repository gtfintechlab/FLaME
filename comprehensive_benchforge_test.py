#!/usr/bin/env python3

"""
Comprehensive E2E Test Suite for BenchForge FLAME Tasks
======================================================

This test suite provides complete coverage of all implemented FLAME tasks in BenchForge,
testing task registration, prompt generation, response extraction, and FLAME compatibility.

Ultra-thinking approach: Systematic validation of every component, edge case testing,
performance analysis, and comprehensive coverage reporting.
"""

import sys
import os
import importlib
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

# Add vendored BenchForge to path
sys.path.append(os.path.join(os.path.dirname(__file__), "vendor/benchforge"))

from bench_forge.tasks.config import PromptFormat
from bench_forge.flame.adapter import FLAMETask


@dataclass
class TestResult:
    """Test result container."""

    task_name: str
    test_category: str
    test_name: str
    status: str  # PASS, FAIL, SKIP
    message: str
    duration: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class TaskTestSummary:
    """Summary of all tests for a specific task."""

    task_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    success_rate: float
    extraction_rate: float
    avg_duration: float
    critical_issues: List[str]


class BenchForgeTestSuite:
    """Comprehensive test suite for BenchForge FLAME tasks."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

        # All discovered FLAME tasks in BenchForge
        self.available_tasks = {
            "fomc": "bench_forge.flame.tasks.fomc",
            "convfinqa": "bench_forge.flame.tasks.convfinqa",
            "finqa": "bench_forge.flame.tasks.finqa",
            "finer": "bench_forge.flame.tasks.finer",
            "finentity": "bench_forge.flame.tasks.finentity",
            "edtsum": "bench_forge.flame.tasks.edtsum",
            "causal_detection": "bench_forge.flame.tasks.causal_detection",
            "causal_classification": "bench_forge.flame.tasks.causal_classification",
            "numclaim": "bench_forge.flame.tasks.numclaim",
            "tatqa": "bench_forge.flame.tasks.tatqa",
            "banking77": "bench_forge.flame.tasks.banking77",
            "ectsum": "bench_forge.flame.tasks.ectsum",
            "finbench": "bench_forge.flame.tasks.finbench",
            "fiqa_sa": "bench_forge.flame.tasks.fiqa_sa",
            "fpb": "bench_forge.flame.tasks.fpb",
            "headlines": "bench_forge.flame.tasks.headlines",
        }

        self.loaded_tasks: Dict[str, Any] = {}
        self.task_instances: Dict[str, FLAMETask] = {}

        print("BenchForge FLAME Tasks - Comprehensive E2E Test Suite")
        print("=" * 60)
        print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Discovered {len(self.available_tasks)} FLAME tasks for testing")
        print()

    def log_result(
        self,
        task_name: str,
        category: str,
        test_name: str,
        status: str,
        message: str,
        duration: float,
        details: Dict[str, Any] = None,
    ):
        """Log a test result."""
        result = TestResult(
            task_name, category, test_name, status, message, duration, details
        )
        self.results.append(result)

        status_icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(
            f"{status_icon} [{task_name}] {category}: {test_name} - {status} ({duration:.3f}s)"
        )
        if status == "FAIL":
            print(f"   Error: {message}")

    def test_task_imports_and_registration(self) -> None:
        """Test 1: Task Import and Registration Validation."""
        print("\n" + "=" * 60)
        print("TEST CATEGORY 1: TASK IMPORTS AND REGISTRATION")
        print("=" * 60)

        for task_name, module_path in self.available_tasks.items():
            start_time = time.time()

            try:
                # Test module import
                module = importlib.import_module(module_path)
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "Import",
                    "Module Import",
                    "PASS",
                    f"Successfully imported {module_path}",
                    duration,
                )
                self.loaded_tasks[task_name] = module

                # Test task class discovery
                start_time = time.time()
                task_class_name = f"{task_name.replace('_', '').title()}Task"

                # Handle special cases
                if task_name == "causal_detection":
                    task_class_name = "CausalDetectionTask"
                elif task_name == "causal_classification":
                    task_class_name = "CausalClassificationTask"
                elif task_name == "fiqa_sa":
                    task_class_name = "FiQASATask"
                elif task_name == "fpb":
                    task_class_name = "FPBTask"

                task_class = getattr(module, task_class_name, None)
                duration = time.time() - start_time

                if task_class:
                    self.log_result(
                        task_name,
                        "Import",
                        "Task Class Discovery",
                        "PASS",
                        f"Found task class {task_class_name}",
                        duration,
                    )

                    # Test task instantiation
                    start_time = time.time()
                    try:
                        config_class_name = (
                            f"{task_name.replace('_', '').title()}Config"
                        )
                        if task_name == "causal_detection":
                            config_class_name = "CausalDetectionConfig"
                        elif task_name == "causal_classification":
                            config_class_name = "CausalClassificationConfig"
                        elif task_name == "fiqa_sa":
                            config_class_name = "FiQASAConfig"
                        elif task_name == "fpb":
                            config_class_name = "FPBConfig"

                        config_class = getattr(module, config_class_name, None)
                        if config_class:
                            config = config_class(name=task_name)
                            task_instance = task_class(config)
                            self.task_instances[task_name] = task_instance
                            duration = time.time() - start_time
                            self.log_result(
                                task_name,
                                "Import",
                                "Task Instantiation",
                                "PASS",
                                "Successfully created task instance",
                                duration,
                            )
                        else:
                            duration = time.time() - start_time
                            self.log_result(
                                task_name,
                                "Import",
                                "Config Class Discovery",
                                "FAIL",
                                f"Config class {config_class_name} not found",
                                duration,
                            )
                    except Exception as e:
                        duration = time.time() - start_time
                        self.log_result(
                            task_name,
                            "Import",
                            "Task Instantiation",
                            "FAIL",
                            f"Failed to instantiate: {str(e)}",
                            duration,
                        )
                else:
                    self.log_result(
                        task_name,
                        "Import",
                        "Task Class Discovery",
                        "FAIL",
                        f"Task class {task_class_name} not found",
                        duration,
                    )

            except Exception as e:
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "Import",
                    "Module Import",
                    "FAIL",
                    f"Import failed: {str(e)}",
                    duration,
                )

    def test_prompt_generation(self) -> None:
        """Test 2: Prompt Generation Across All Formats."""
        print("\n" + "=" * 60)
        print("TEST CATEGORY 2: PROMPT GENERATION")
        print("=" * 60)

        test_samples = self._get_test_samples()
        prompt_formats = [
            PromptFormat.ZERO_SHOT,
            PromptFormat.FEW_SHOT,
            PromptFormat.CHAIN_OF_THOUGHT,
        ]

        for task_name, task_instance in self.task_instances.items():
            sample = test_samples.get(task_name, {})

            for format_type in prompt_formats:
                start_time = time.time()
                try:
                    prompt = task_instance.create_prompt(sample, format_type)
                    duration = time.time() - start_time

                    if prompt and len(prompt) > 50:  # Reasonable prompt length
                        self.log_result(
                            task_name,
                            "Prompts",
                            f"{format_type.value} Format",
                            "PASS",
                            f"Generated prompt ({len(prompt)} chars)",
                            duration,
                            {"prompt_length": len(prompt), "format": format_type.value},
                        )
                    else:
                        self.log_result(
                            task_name,
                            "Prompts",
                            f"{format_type.value} Format",
                            "FAIL",
                            f"Prompt too short or empty ({len(prompt) if prompt else 0} chars)",
                            duration,
                        )
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_result(
                        task_name,
                        "Prompts",
                        f"{format_type.value} Format",
                        "FAIL",
                        f"Prompt generation failed: {str(e)}",
                        duration,
                    )

    def test_response_extraction(self) -> None:
        """Test 3: Response Extraction with Diverse Inputs."""
        print("\n" + "=" * 60)
        print("TEST CATEGORY 3: RESPONSE EXTRACTION")
        print("=" * 60)

        test_responses = self._get_test_responses()

        for task_name, task_instance in self.task_instances.items():
            responses = test_responses.get(task_name, test_responses["generic"])
            successful_extractions = 0
            total_responses = len(responses)

            for i, response_text in enumerate(responses):
                start_time = time.time()
                try:
                    extracted = task_instance.extract_response(response_text)
                    duration = time.time() - start_time

                    if extracted is not None:
                        successful_extractions += 1
                        self.log_result(
                            task_name,
                            "Extraction",
                            f"Response {i+1}",
                            "PASS",
                            f"Extracted: {str(extracted)[:50]}",
                            duration,
                        )
                    else:
                        self.log_result(
                            task_name,
                            "Extraction",
                            f"Response {i+1}",
                            "FAIL",
                            f"No extraction from: {response_text[:50]}...",
                            duration,
                        )
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_result(
                        task_name,
                        "Extraction",
                        f"Response {i+1}",
                        "FAIL",
                        f"Extraction error: {str(e)}",
                        duration,
                    )

            # Overall extraction rate test
            extraction_rate = (
                (successful_extractions / total_responses) * 100
                if total_responses > 0
                else 0
            )
            status = (
                "PASS" if extraction_rate >= 70 else "FAIL"
            )  # 70% threshold for success
            self.log_result(
                task_name,
                "Extraction",
                "Overall Rate",
                status,
                f"Extraction rate: {extraction_rate:.1f}% ({successful_extractions}/{total_responses})",
                0.0,
                {"extraction_rate": extraction_rate},
            )

    def test_flame_compatibility(self) -> None:
        """Test 4: FLAME Compatibility and Data Formatting."""
        print("\n" + "=" * 60)
        print("TEST CATEGORY 4: FLAME COMPATIBILITY")
        print("=" * 60)

        test_samples = self._get_test_samples()

        for task_name, task_instance in self.task_instances.items():
            sample = test_samples.get(task_name, {})

            # Test format_results method
            start_time = time.time()
            try:
                # Mock data for testing format_results
                samples = [sample] * 3
                prompts = [task_instance.create_prompt(sample)] * 3
                raw_responses = [
                    "Test response 1",
                    "Test response 2",
                    "Test response 3",
                ]
                extracted = [
                    task_instance.extract_response(resp, sample)
                    for resp in raw_responses
                ]

                results_df = task_instance.format_results(
                    samples, prompts, raw_responses, extracted
                )
                duration = time.time() - start_time

                # Check if DataFrame has required FLAME columns
                required_columns = self._get_required_flame_columns(task_name)
                missing_columns = []
                for col in required_columns:
                    if col not in results_df.columns:
                        missing_columns.append(col)

                if not missing_columns and len(results_df) == 3:
                    self.log_result(
                        task_name,
                        "FLAME",
                        "Results Formatting",
                        "PASS",
                        f"DataFrame created with {len(results_df.columns)} columns",
                        duration,
                        {"columns": list(results_df.columns), "rows": len(results_df)},
                    )
                else:
                    self.log_result(
                        task_name,
                        "FLAME",
                        "Results Formatting",
                        "FAIL",
                        f"Missing columns: {missing_columns} or wrong row count",
                        duration,
                    )

            except Exception as e:
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "FLAME",
                    "Results Formatting",
                    "FAIL",
                    f"format_results failed: {str(e)}",
                    duration,
                )

            # Test ground truth extraction
            start_time = time.time()
            try:
                ground_truth = task_instance.get_ground_truth(sample)
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "FLAME",
                    "Ground Truth",
                    "PASS",
                    f"Extracted: {str(ground_truth)[:50]}",
                    duration,
                )
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "FLAME",
                    "Ground Truth",
                    "FAIL",
                    f"Ground truth extraction failed: {str(e)}",
                    duration,
                )

    def test_performance_and_integration(self) -> None:
        """Test 5: Performance and Integration Testing."""
        print("\n" + "=" * 60)
        print("TEST CATEGORY 5: PERFORMANCE AND INTEGRATION")
        print("=" * 60)

        for task_name, task_instance in self.task_instances.items():
            # Test task configuration
            start_time = time.time()
            try:
                config = task_instance.config
                duration = time.time() - start_time

                required_attrs = ["name", "prompt_format", "huggingface_dataset"]
                missing_attrs = [
                    attr for attr in required_attrs if not hasattr(config, attr)
                ]

                if not missing_attrs:
                    self.log_result(
                        task_name,
                        "Performance",
                        "Configuration",
                        "PASS",
                        "All required config attributes present",
                        duration,
                    )
                else:
                    self.log_result(
                        task_name,
                        "Performance",
                        "Configuration",
                        "FAIL",
                        f"Missing config attributes: {missing_attrs}",
                        duration,
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "Performance",
                    "Configuration",
                    "FAIL",
                    f"Config check failed: {str(e)}",
                    duration,
                )

            # Test task statistics tracking
            start_time = time.time()
            try:
                initial_stats = dict(task_instance._stats)

                # Perform operations to update stats
                sample = self._get_test_samples().get(task_name, {})
                task_instance.create_prompt(sample)
                task_instance.extract_response("Test response")

                updated_stats = dict(task_instance._stats)
                duration = time.time() - start_time

                stats_updated = any(
                    updated_stats[key] != initial_stats.get(key, 0)
                    for key in updated_stats
                )

                if stats_updated:
                    self.log_result(
                        task_name,
                        "Performance",
                        "Statistics Tracking",
                        "PASS",
                        "Stats updated correctly",
                        duration,
                        {"stats": updated_stats},
                    )
                else:
                    self.log_result(
                        task_name,
                        "Performance",
                        "Statistics Tracking",
                        "FAIL",
                        "Stats not updated",
                        duration,
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_result(
                    task_name,
                    "Performance",
                    "Statistics Tracking",
                    "FAIL",
                    f"Stats tracking failed: {str(e)}",
                    duration,
                )

    def _get_test_samples(self) -> Dict[str, Dict[str, Any]]:
        """Get test samples for each task type."""
        return {
            "fomc": {
                "sentence": "The Federal Reserve announced an interest rate increase of 0.25%",
                "label": "hawkish",
            },
            "fpb": {
                "sentence": "The company reported strong quarterly earnings with revenue growth",
                "label": "positive",
            },
            "convfinqa": {
                "pre_text": "Financial data shows",
                "post_text": "revenue increased",
                "table": [["Q1", "100"], ["Q2", "120"]],
                "qa": {"question": "What was Q2 revenue?", "exe_ans": "120"},
            },
            "finqa": {
                "pre_text": "Company financials",
                "post_text": "show growth",
                "table": [["Revenue", "100M"], ["Profit", "20M"]],
                "qa": {"question": "What is the profit margin?", "exe_ans": "20%"},
            },
            "finer": {
                "sentence": "Apple Inc. reported earnings of $5 billion",
                "label": ["B-ORG", "I-ORG", "O", "O", "O", "B-MONEY", "I-MONEY"],
            },
            "finentity": {
                "sentence": "Tesla stock rose 5%",
                "entity": "Tesla",
                "label": "positive",
            },
            "edtsum": {
                "text": "Company earnings report shows strong performance with revenue up 15%",
                "summary": "Revenue increased 15%",
            },
            "causal_detection": {
                "sentence": "Rate increase caused market decline",
                "label": ["O", "O", "B-CAUSE", "O", "B-EFFECT"],
            },
            "causal_classification": {
                "sentence": "Higher rates led to market volatility",
                "label": 2,
            },
            "numclaim": {"sentence": "The stock rose 15% today", "label": 1},
            "tatqa": {
                "table": [["Q1", "100"], ["Q2", "120"]],
                "paragraphs": ["Revenue grew"],
                "question": "What was Q2 revenue?",
                "answer": "120",
            },
            "banking77": {
                "text": "I want to activate my credit card",
                "label": "activate_my_card",
            },
            "ectsum": {
                "context": "Q3 earnings call: Revenue $2.5B, up 15% YoY. Net income $400M.",
                "response": "• Q3 revenue $2.5B, +15% YoY\n• Net income $400M",
            },
            "finbench": {
                "X_profile": "Age: 35, Income: $75000, Credit Score: 720",
                "y": "LOW RISK",
            },
            "fiqa_sa": {
                "sentence": "The stock performance was excellent this quarter",
                "target": "stock",
                "label": 0.8,
            },
            "headlines": {
                "headline": "Apple stock rises 5% on strong earnings",
                "label": {
                    "Price_or_Not": 1,
                    "Direction_Up": 1,
                    "Direction_Down": 0,
                    "Direction_Constant": 0,
                    "Past_Price": 0,
                    "Future_Price": 0,
                    "Past_News": 1,
                },
            },
        }

    def _get_test_responses(self) -> Dict[str, List[str]]:
        """Get test responses for extraction testing."""
        return {
            "generic": [
                "The answer is positive",
                "Classification: negative",
                "Result: neutral",
                "I think this is positive based on the analysis",
                "",  # Empty response
                "This is ambiguous and unclear",
                "Answer: 42",
                "LOW RISK",
                "HIGH RISK",
                "positive sentiment detected",
            ],
            "fomc": [
                "hawkish",
                "The sentiment is dovish",
                "Classification: neutral",
                "This statement shows a hawkish stance",
                "dovish sentiment detected",
            ],
            "banking77": [
                "activate_my_card",
                "This is about card activation",
                "Category: activate_my_card",
                "The intent is to activate a card",
                "apple_pay_or_google_pay",
            ],
            "finbench": [
                "LOW RISK",
                "HIGH RISK",
                "This applicant is low risk",
                "Classification: HIGH RISK",
                "The risk level is low",
            ],
            "ectsum": [
                "• Revenue increased 15%\n• Profit margins improved",
                "Key points:\n- Strong quarterly performance\n- Revenue growth of 15%",
                "Summary: Company performed well with revenue up 15%",
                "• Q3 revenue $2.5B\n• Net income $400M",
            ],
        }

    def _get_required_flame_columns(self, task_name: str) -> List[str]:
        """Get required FLAME columns for each task type."""
        base_columns = ["prompt", "raw_response", "extracted_response", "ground_truth"]

        task_specific = {
            "fomc": ["sentence", "label", "llm_responses", "actual_labels"],
            "fpb": ["sentence", "label", "llm_responses", "actual_labels"],
            "finbench": ["X_profile", "y", "llm_responses", "extracted_labels"],
            "ectsum": [
                "documents",
                "llm_responses",
                "actual_labels",
                "extracted_labels",
            ],
            "banking77": ["text", "label", "llm_responses", "extracted_labels"],
        }

        return base_columns + task_specific.get(
            task_name, ["input", "label", "llm_responses"]
        )

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive test coverage report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST COVERAGE REPORT")
        print("=" * 60)

        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        skipped_tests = sum(1 for r in self.results if r.status == "SKIP")

        overall_success_rate = (
            (passed_tests / total_tests * 100) if total_tests > 0 else 0
        )
        total_duration = time.time() - self.start_time

        print("\nOVERALL TEST RESULTS:")
        print(f"Total Tests Executed: {total_tests}")
        print(f"✓ Passed: {passed_tests}")
        print(f"✗ Failed: {failed_tests}")
        print(f"⚠ Skipped: {skipped_tests}")
        print(f"Success Rate: {overall_success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")

        # Per-task analysis
        task_summaries = self._generate_task_summaries()

        print("\nPER-TASK ANALYSIS:")
        print("-" * 60)

        for task_name, summary in task_summaries.items():
            print(f"\n{task_name.upper()}:")
            print(
                f"  Tests: {summary.passed_tests}/{summary.total_tests} passed ({summary.success_rate:.1f}%)"
            )
            if hasattr(summary, "extraction_rate"):
                print(f"  Extraction Rate: {summary.extraction_rate:.1f}%")
            print(f"  Avg Duration: {summary.avg_duration:.3f}s")
            if summary.critical_issues:
                print(f"  Critical Issues: {len(summary.critical_issues)}")
                for issue in summary.critical_issues[:3]:  # Show top 3
                    print(f"    - {issue}")

        # Category analysis
        print("\nTEST CATEGORY ANALYSIS:")
        print("-" * 60)
        categories = {}
        for result in self.results:
            if result.test_category not in categories:
                categories[result.test_category] = {"total": 0, "passed": 0}
            categories[result.test_category]["total"] += 1
            if result.status == "PASS":
                categories[result.test_category]["passed"] += 1

        for category, stats in categories.items():
            success_rate = (
                (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            print(
                f"{category}: {stats['passed']}/{stats['total']} passed ({success_rate:.1f}%)"
            )

        # Critical findings
        critical_failures = [
            r
            for r in self.results
            if r.status == "FAIL"
            and any(
                keyword in r.test_name.lower()
                for keyword in ["import", "instantiation", "registration"]
            )
        ]

        if critical_failures:
            print("\nCRITICAL FAILURES (affecting core functionality):")
            print("-" * 60)
            for failure in critical_failures:
                print(
                    f"[{failure.task_name}] {failure.test_category}: {failure.test_name}"
                )
                print(f"  Error: {failure.message}")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 60)

        if overall_success_rate >= 90:
            print(
                "✓ Excellent test coverage. BenchForge FLAME tasks are production-ready."
            )
        elif overall_success_rate >= 75:
            print(
                "⚠ Good test coverage. Minor issues need attention before production."
            )
        else:
            print("✗ Poor test coverage. Significant issues require resolution.")

        # Specific recommendations based on failures
        extraction_failures = [
            r
            for r in self.results
            if r.status == "FAIL" and "extraction" in r.test_name.lower()
        ]
        if extraction_failures:
            print("- Improve response extraction strategies for better robustness")

        prompt_failures = [
            r
            for r in self.results
            if r.status == "FAIL" and "prompt" in r.test_name.lower()
        ]
        if prompt_failures:
            print("- Review prompt generation for failed tasks")

        import_failures = [
            r
            for r in self.results
            if r.status == "FAIL" and "import" in r.test_name.lower()
        ]
        if import_failures:
            print("- Fix import/registration issues for failed tasks")

        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _generate_task_summaries(self) -> Dict[str, TaskTestSummary]:
        """Generate per-task summaries."""
        summaries = {}

        for task_name in self.available_tasks.keys():
            task_results = [r for r in self.results if r.task_name == task_name]

            if not task_results:
                continue

            total = len(task_results)
            passed = sum(1 for r in task_results if r.status == "PASS")
            failed = sum(1 for r in task_results if r.status == "FAIL")
            skipped = sum(1 for r in task_results if r.status == "SKIP")

            success_rate = (passed / total * 100) if total > 0 else 0
            avg_duration = (
                sum(r.duration for r in task_results) / total if total > 0 else 0
            )

            # Calculate extraction rate if available
            extraction_results = [
                r
                for r in task_results
                if "extraction" in r.test_name.lower() and r.details
            ]
            extraction_rate = 0.0
            if extraction_results:
                for result in extraction_results:
                    if result.details and "extraction_rate" in result.details:
                        extraction_rate = result.details["extraction_rate"]
                        break

            critical_issues = [
                r.message
                for r in task_results
                if r.status == "FAIL"
                and any(
                    keyword in r.test_name.lower()
                    for keyword in ["import", "instantiation", "format"]
                )
            ]

            summaries[task_name] = TaskTestSummary(
                task_name=task_name,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                success_rate=success_rate,
                extraction_rate=extraction_rate,
                avg_duration=avg_duration,
                critical_issues=critical_issues,
            )

        return summaries

    def run_comprehensive_tests(self) -> None:
        """Run all test categories."""
        try:
            self.test_task_imports_and_registration()
            self.test_prompt_generation()
            self.test_response_extraction()
            self.test_flame_compatibility()
            self.test_performance_and_integration()
        except Exception as e:
            print(f"\nCRITICAL ERROR during test execution: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
        finally:
            self.generate_comprehensive_report()


def main():
    """Main test execution."""
    test_suite = BenchForgeTestSuite()
    test_suite.run_comprehensive_tests()


if __name__ == "__main__":
    main()
