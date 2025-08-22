#!/usr/bin/env python3

"""
Focused E2E Test Suite for Key BenchForge FLAME Tasks
===================================================

Ultra-thinking focused test on the most critical and recently implemented tasks:
- TATQA, Banking77, ECTSum, FinBench (recently implemented)
- FPB, FiQA-SA, Headlines (core tasks)
- Banking77 (complex classification)

This provides deep validation of extraction rates, FLAME compatibility, and performance.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "vendor/benchforge"))

from bench_forge.tasks.config import PromptFormat
from bench_forge.flame.tasks.tatqa import TATQATask, TATQAConfig
from bench_forge.flame.tasks.banking77 import Banking77Task, Banking77Config
from bench_forge.flame.tasks.ectsum import ECTSumTask, ECTSumConfig
from bench_forge.flame.tasks.finbench import FinBenchTask, FinBenchConfig
from bench_forge.flame.tasks.fpb import FPBTask, FPBConfig
from bench_forge.flame.tasks.fiqa_sa import FiQASATask, FiQASAConfig
from bench_forge.flame.tasks.headlines import HeadlinesTask, HeadlinesConfig


def test_task_comprehensive(
    task_class, config_class, task_name, test_samples, test_responses
):
    """Comprehensive test for a single task."""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST: {task_name.upper()}")
    print(f"{'='*60}")

    results = {
        "initialization": False,
        "prompt_generation": {"zero_shot": False, "few_shot": False, "cot": False},
        "extraction_rate": 0.0,
        "flame_compatibility": False,
        "performance": False,
        "issues": [],
    }

    try:
        # 1. Task Initialization
        print(f"[{task_name}] Testing initialization...")
        config = config_class(name=task_name)
        task = task_class(config)
        results["initialization"] = True
        print(f"✓ [{task_name}] Initialization successful")

        # 2. Prompt Generation Tests
        print(f"[{task_name}] Testing prompt generation...")
        sample = test_samples.get(task_name, {})

        for format_name, format_type in [
            ("zero_shot", PromptFormat.ZERO_SHOT),
            ("few_shot", PromptFormat.FEW_SHOT),
            ("cot", PromptFormat.CHAIN_OF_THOUGHT),
        ]:
            try:
                prompt = task.create_prompt(sample, format_type)
                if prompt and len(prompt) > 50:
                    results["prompt_generation"][format_name] = True
                    print(f"✓ [{task_name}] {format_name} prompt: {len(prompt)} chars")
                else:
                    print(f"✗ [{task_name}] {format_name} prompt too short")
            except Exception as e:
                print(f"✗ [{task_name}] {format_name} prompt failed: {str(e)}")
                results["issues"].append(f"Prompt {format_name}: {str(e)}")

        # 3. Response Extraction Tests
        print(f"[{task_name}] Testing response extraction...")
        responses = test_responses.get(task_name, [])
        successful = 0
        total = len(responses)

        for i, response in enumerate(responses):
            try:
                extracted = task.extract_response(response, sample)
                if extracted is not None:
                    successful += 1
                    print(f"✓ [{task_name}] Response {i+1}: {str(extracted)[:50]}")
                else:
                    print(f"✗ [{task_name}] Response {i+1}: No extraction")
            except Exception as e:
                print(f"✗ [{task_name}] Response {i+1}: Error - {str(e)}")
                results["issues"].append(f"Extraction {i+1}: {str(e)}")

        extraction_rate = (successful / total * 100) if total > 0 else 0
        results["extraction_rate"] = extraction_rate
        print(
            f"[{task_name}] Extraction rate: {extraction_rate:.1f}% ({successful}/{total})"
        )

        # 4. FLAME Compatibility Tests
        print(f"[{task_name}] Testing FLAME compatibility...")
        try:
            # Test format_results
            samples = [sample] * 3
            prompts = [task.create_prompt(sample)] * 3
            raw_responses = (
                responses[:3]
                if len(responses) >= 3
                else responses + [""] * (3 - len(responses))
            )
            extracted_responses = [
                task.extract_response(resp, sample) for resp in raw_responses
            ]

            df = task.format_results(
                samples, prompts, raw_responses, extracted_responses
            )

            # Check basic DataFrame properties
            if len(df) == 3 and len(df.columns) >= 5:
                results["flame_compatibility"] = True
                print(
                    f"✓ [{task_name}] FLAME format: {len(df)} rows, {len(df.columns)} columns"
                )

                # Check for key FLAME columns
                flame_columns = ["prompt", "input", "ground_truth"]
                missing = [col for col in flame_columns if col not in df.columns]
                if missing:
                    print(f"⚠ [{task_name}] Missing standard columns: {missing}")
                else:
                    print(f"✓ [{task_name}] All standard FLAME columns present")
            else:
                print(
                    f"✗ [{task_name}] FLAME format issues: {len(df)} rows, {len(df.columns)} columns"
                )
                results["issues"].append("FLAME format validation failed")

        except Exception as e:
            print(f"✗ [{task_name}] FLAME compatibility failed: {str(e)}")
            results["issues"].append(f"FLAME compatibility: {str(e)}")

        # 5. Performance Tests
        print(f"[{task_name}] Testing performance...")
        try:
            # Test statistics tracking
            initial_stats = dict(task._stats) if hasattr(task, "_stats") else {}
            task.create_prompt(sample)
            task.extract_response("test response")
            updated_stats = dict(task._stats) if hasattr(task, "_stats") else {}

            if updated_stats != initial_stats:
                results["performance"] = True
                print(f"✓ [{task_name}] Statistics tracking working")
            else:
                print(f"✗ [{task_name}] Statistics not updating")
                results["issues"].append("Statistics tracking not working")

        except Exception as e:
            print(f"✗ [{task_name}] Performance test failed: {str(e)}")
            results["issues"].append(f"Performance: {str(e)}")

    except Exception as e:
        print(f"✗ [{task_name}] Critical failure: {str(e)}")
        results["issues"].append(f"Critical: {str(e)}")

    return results


def main():
    """Run focused comprehensive tests."""
    print("BenchForge FLAME Tasks - Focused E2E Test Suite")
    print("=" * 60)
    print("Testing key implementations with task-specific test data")
    print()

    # Task configurations
    test_configs = [
        (TATQATask, TATQAConfig, "tatqa"),
        (Banking77Task, Banking77Config, "banking77"),
        (ECTSumTask, ECTSumConfig, "ectsum"),
        (FinBenchTask, FinBenchConfig, "finbench"),
        (FPBTask, FPBConfig, "fpb"),
        (FiQASATask, FiQASAConfig, "fiqa_sa"),
        (HeadlinesTask, HeadlinesConfig, "headlines"),
    ]

    # Task-specific test samples
    test_samples = {
        "tatqa": {
            "table": [
                ["Q1 Revenue", "100M"],
                ["Q2 Revenue", "120M"],
                ["Growth", "20%"],
            ],
            "paragraphs": [
                "The company showed strong performance in Q2 with revenue increasing from 100M to 120M."
            ],
            "question": "What was the revenue growth from Q1 to Q2?",
            "answer": "20%",
        },
        "banking77": {
            "text": "I need to activate my new credit card that just arrived",
            "label": "activate_my_card",
        },
        "ectsum": {
            "context": "Good morning and welcome to our Q3 2023 earnings call. I am pleased to report that we achieved record quarterly revenue of $2.5 billion, representing a 15% increase year-over-year. Our net income was $400 million, or $2.10 per share, which beat analyst estimates of $1.95 per share. Looking ahead, we are raising our full-year guidance.",
            "response": "• Q3 revenue $2.5B, +15% YoY\n• Net income $400M, $2.10/share\n• Beat estimates of $1.95/share\n• Raising full-year guidance",
        },
        "finbench": {
            "X_profile": "Age: 35, Annual Income: $75000, Total Debt: $12000, Credit Score: 720, Employment Status: Full-time, Education: Bachelor's Degree",
            "y": "LOW RISK",
        },
        "fpb": {
            "sentence": "The company reported excellent quarterly earnings with strong revenue growth and improved margins.",
            "label": "positive",
        },
        "fiqa_sa": {
            "sentence": "The stock performance was outstanding this quarter with significant gains.",
            "target": "stock",
            "label": 0.8,
        },
        "headlines": {
            "headline": "Apple stock rises 5% on better-than-expected quarterly earnings report",
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

    # Task-specific test responses
    test_responses = {
        "tatqa": [
            "20%",
            "The revenue growth from Q1 to Q2 was 20%",
            "Answer: 20%",
            "Based on the table, the growth was 20%",
            "(120M - 100M) / 100M = 20%",
            "Twenty percent increase",
            "Growth rate: 20%",
        ],
        "banking77": [
            "activate_my_card",
            "This is about card activation",
            "The customer wants to activate_my_card",
            "Intent: activate_my_card",
            "Card activation request - activate_my_card",
            "activate my card",
            "category: activate_my_card",
        ],
        "ectsum": [
            "• Q3 revenue $2.5B, +15% YoY\n• Net income $400M",
            "Key points:\n- Record Q3 revenue of $2.5 billion\n- 15% year-over-year growth\n- Beat analyst estimates",
            "• Revenue: $2.5B (+15%)\n• EPS: $2.10 vs $1.95 estimate\n• Raised guidance",
            "Summary: Strong Q3 with $2.5B revenue, beating estimates",
            "• Record quarterly revenue $2.5B\n• Net income $400M\n• Guidance raised",
        ],
        "finbench": [
            "LOW RISK",
            "This applicant is low risk for loan approval",
            "Classification: LOW RISK",
            "Risk Assessment: LOW RISK\nGood credit score and stable employment",
            "APPROVE - LOW RISK candidate",
            "The risk category is LOW RISK",
            "LOW RISK\nStrong financial profile with stable income",
        ],
        "fpb": [
            "positive",
            "The sentiment is positive",
            "Classification: positive",
            "This shows positive sentiment about earnings",
            "Positive financial sentiment detected",
            "positive sentiment",
            "Result: positive",
        ],
        "fiqa_sa": [
            "0.8",
            "The sentiment score is 0.8",
            "Score: 0.8 (positive)",
            "Sentiment: 0.8",
            "Rating: 0.8/1.0",
            "Very positive: 0.8",
            "Sentiment score: 0.8",
        ],
        "headlines": [
            "Price_or_Not: 1, Direction_Up: 1, Direction_Down: 0, Direction_Constant: 0, Past_Price: 0, Future_Price: 0, Past_News: 1",
            '{"Price_or_Not": 1, "Direction_Up": 1, "Direction_Down": 0, "Direction_Constant": 0, "Past_Price": 0, "Future_Price": 0, "Past_News": 1}',
            "1,1,0,0,0,0,1",
            "Price mentioned: Yes, Direction: Up, Past news: Yes",
            "1 1 0 0 0 0 1",
            "Price_or_Not=1; Direction_Up=1; Past_News=1; others=0",
        ],
    }

    # Run comprehensive tests
    all_results = {}
    for task_class, config_class, task_name in test_configs:
        results = test_task_comprehensive(
            task_class, config_class, task_name, test_samples, test_responses
        )
        all_results[task_name] = results

    # Generate summary report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")

    total_tasks = len(all_results)
    successful_init = sum(1 for r in all_results.values() if r["initialization"])
    avg_extraction = (
        sum(r["extraction_rate"] for r in all_results.values()) / total_tasks
        if total_tasks > 0
        else 0
    )
    flame_compatible = sum(1 for r in all_results.values() if r["flame_compatibility"])

    print("\nOVERALL METRICS:")
    print(f"Total Tasks Tested: {total_tasks}")
    print(
        f"✓ Successful Initialization: {successful_init}/{total_tasks} ({successful_init/total_tasks*100:.1f}%)"
    )
    print(
        f"✓ FLAME Compatible: {flame_compatible}/{total_tasks} ({flame_compatible/total_tasks*100:.1f}%)"
    )
    print(f"📊 Average Extraction Rate: {avg_extraction:.1f}%")

    print("\nPER-TASK BREAKDOWN:")
    print("-" * 60)

    for task_name, results in all_results.items():
        status = (
            "✓"
            if results["initialization"] and results["extraction_rate"] >= 70
            else "⚠"
            if results["initialization"]
            else "✗"
        )
        print(
            f"{status} {task_name.upper():12} | Init: {'✓' if results['initialization'] else '✗'} | "
            f"Extract: {results['extraction_rate']:5.1f}% | "
            f"FLAME: {'✓' if results['flame_compatibility'] else '✗'} | "
            f"Issues: {len(results['issues'])}"
        )

        if results["issues"]:
            for issue in results["issues"][:2]:  # Show top 2 issues
                print(f"    → {issue}")

    print("\nHIGH-PERFORMING TASKS:")
    high_performers = [
        name
        for name, r in all_results.items()
        if r["initialization"]
        and r["extraction_rate"] >= 80
        and r["flame_compatibility"]
    ]

    if high_performers:
        for task in high_performers:
            rate = all_results[task]["extraction_rate"]
            print(f"✓ {task}: {rate:.1f}% extraction rate")
    else:
        print(
            "No tasks meet high-performance criteria (80%+ extraction, FLAME compatible)"
        )

    print("\nRECOMMENDATIONS:")
    if avg_extraction >= 75:
        print("✓ Strong extraction performance across tasks")
    elif avg_extraction >= 50:
        print("⚠ Moderate extraction performance - review response patterns")
    else:
        print("✗ Low extraction performance - significant improvements needed")

    if flame_compatible >= total_tasks * 0.8:
        print("✓ Good FLAME compatibility across tasks")
    else:
        print("⚠ FLAME compatibility issues need addressing")

    print(
        f"\nBenchForge FLAME Tasks Status: {'PRODUCTION READY' if avg_extraction >= 70 and flame_compatible >= total_tasks * 0.8 else 'NEEDS IMPROVEMENT'}"
    )


if __name__ == "__main__":
    main()
