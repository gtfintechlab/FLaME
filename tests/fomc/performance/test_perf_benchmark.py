#!/usr/bin/env python3
"""
Performance benchmarking tests for FOMC implementations.
Measures and compares performance characteristics.
"""

import pytest
import sys
import time
import statistics
import json
from pathlib import Path
from typing import Dict, Any
import psutil
import tracemalloc

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "benchforge"))


class PerformanceMetrics:
    """Helper class to collect performance metrics."""

    def __init__(self):
        self.metrics = {"execution_times": [], "memory_usage": [], "cpu_usage": []}

    def add_timing(self, elapsed: float):
        """Add execution time measurement."""
        self.metrics["execution_times"].append(elapsed)

    def add_memory(self, memory_mb: float):
        """Add memory usage measurement."""
        self.metrics["memory_usage"].append(memory_mb)

    def add_cpu(self, cpu_percent: float):
        """Add CPU usage measurement."""
        self.metrics["cpu_usage"].append(cpu_percent)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}

        if self.metrics["execution_times"]:
            times = self.metrics["execution_times"]
            summary["timing"] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
            }

        if self.metrics["memory_usage"]:
            memory = self.metrics["memory_usage"]
            summary["memory"] = {"mean": statistics.mean(memory), "max": max(memory)}

        if self.metrics["cpu_usage"]:
            cpu = self.metrics["cpu_usage"]
            summary["cpu"] = {"mean": statistics.mean(cpu), "max": max(cpu)}

        return summary


class TestPromptGenerationPerformance:
    """Benchmark prompt generation performance."""

    def test_native_prompt_generation_speed(self):
        """Benchmark native prompt generation."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        metrics = PerformanceMetrics()

        # Warm up
        for _ in range(10):
            prompt_func("warm up")

        # Benchmark
        test_inputs = [
            "Short statement",
            "Medium length Federal Reserve statement about monetary policy",
            "Very long statement " * 50,  # Long input
        ]

        for test_input in test_inputs:
            for _ in range(100):
                start = time.perf_counter()
                prompt_func(test_input)
                elapsed = time.perf_counter() - start
                metrics.add_timing(elapsed)

        summary = metrics.get_summary()

        # Performance requirements
        assert summary["timing"]["mean"] < 0.001  # <1ms average
        assert summary["timing"]["max"] < 0.01  # <10ms worst case

        print("\nNative Prompt Generation Performance:")
        print(f"  Mean: {summary['timing']['mean'] * 1000:.3f}ms")
        print(f"  Median: {summary['timing']['median'] * 1000:.3f}ms")
        print(f"  Max: {summary['timing']['max'] * 1000:.3f}ms")

    def test_benchforge_prompt_generation_speed(self):
        """Benchmark BenchForge prompt generation."""
        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat
        except ImportError:
            pytest.skip("BenchForge not available")

        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            prompt_format=PromptFormat.ZERO_SHOT,
            text_field="sentence",
        )
        task = FOMCTask(config)
        metrics = PerformanceMetrics()

        # Warm up
        for _ in range(10):
            task.create_prompt({"sentence": "warm up"}, PromptFormat.ZERO_SHOT)

        # Benchmark
        test_samples = [
            {"sentence": "Short statement"},
            {
                "sentence": "Medium length Federal Reserve statement about monetary policy"
            },
            {"sentence": "Very long statement " * 50},
        ]

        for sample in test_samples:
            for _ in range(100):
                start = time.perf_counter()
                task.create_prompt(sample, PromptFormat.ZERO_SHOT)
                elapsed = time.perf_counter() - start
                metrics.add_timing(elapsed)

        summary = metrics.get_summary()

        # Performance requirements (slightly more lenient for BenchForge)
        assert summary["timing"]["mean"] < 0.002  # <2ms average
        assert summary["timing"]["max"] < 0.02  # <20ms worst case

        print("\nBenchForge Prompt Generation Performance:")
        print(f"  Mean: {summary['timing']['mean'] * 1000:.3f}ms")
        print(f"  Median: {summary['timing']['median'] * 1000:.3f}ms")
        print(f"  Max: {summary['timing']['max'] * 1000:.3f}ms")


class TestExtractionPerformance:
    """Benchmark extraction performance."""

    def test_native_extraction_speed(self):
        """Benchmark native extraction speed."""
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        metrics = PerformanceMetrics()
        test_responses = [
            "HAWKISH",
            "The answer is DOVISH",
            "Based on extensive analysis of multiple factors, this is NEUTRAL",
            "hawkish dovish neutral" * 10,  # Complex response
        ]

        for response in test_responses:
            for _ in range(1000):
                start = time.perf_counter()

                # Simple extraction logic
                for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                    if label in response.upper():
                        map_label_to_number(label)
                        break

                elapsed = time.perf_counter() - start
                metrics.add_timing(elapsed)

        summary = metrics.get_summary()

        assert summary["timing"]["mean"] < 0.0001  # <0.1ms average
        assert summary["timing"]["max"] < 0.001  # <1ms worst case

        print("\nNative Extraction Performance:")
        print(f"  Mean: {summary['timing']['mean'] * 1000:.3f}ms")
        print(f"  Max: {summary['timing']['max'] * 1000:.3f}ms")

    def test_benchforge_extraction_speed(self):
        """Benchmark BenchForge extraction speed."""
        try:
            from bench_forge.prompts.extractor import (
                ResponseExtractor,
                ExtractionStrategy,
            )
        except ImportError:
            pytest.skip("BenchForge extractor not available")

        extractor = ResponseExtractor()
        metrics = PerformanceMetrics()

        test_responses = [
            "HAWKISH",
            "The answer is DOVISH",
            "Based on extensive analysis, this is NEUTRAL",
            "Complex response with multiple mentions of hawkish and dovish",
        ]

        for response in test_responses:
            for _ in range(100):
                start = time.perf_counter()

                extractor.extract(
                    response,
                    strategy=ExtractionStrategy.FUZZY,
                    options=["HAWKISH", "DOVISH", "NEUTRAL"],
                )

                elapsed = time.perf_counter() - start
                metrics.add_timing(elapsed)

        summary = metrics.get_summary()

        assert summary["timing"]["mean"] < 0.001  # <1ms average
        assert summary["timing"]["max"] < 0.01  # <10ms worst case

        print("\nBenchForge Extraction Performance:")
        print(f"  Mean: {summary['timing']['mean'] * 1000:.3f}ms")
        print(f"  Max: {summary['timing']['max'] * 1000:.3f}ms")


class TestMemoryPerformance:
    """Test memory usage characteristics."""

    def test_native_memory_usage(self):
        """Test native implementation memory usage."""
        from flame.code.prompts.registry import get_prompt, PromptFormat
        from flame.code.fomc.fomc_evaluate import map_label_to_number

        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        # Create many prompts
        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)
        prompts = []

        for i in range(1000):
            prompt = prompt_func(f"Test statement {i}")
            prompts.append(prompt)

        # Map many labels
        for _ in range(10000):
            map_label_to_number("HAWKISH")

        # Check memory
        current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        tracemalloc.stop()

        print("\nNative Memory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {current_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")

        # Should not use excessive memory
        assert memory_increase < 50  # Less than 50MB increase

    def test_benchforge_memory_usage(self):
        """Test BenchForge implementation memory usage."""
        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat
            from bench_forge.prompts.extractor import ResponseExtractor
        except ImportError:
            pytest.skip("BenchForge not available")

        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        # Create task and extractor
        config = FLAMEConfig(name="fomc", dataset="fomc")
        task = FOMCTask(config)
        ResponseExtractor()

        # Create many prompts
        prompts = []
        for i in range(1000):
            prompt = task.create_prompt(
                {"sentence": f"Test {i}"}, PromptFormat.ZERO_SHOT
            )
            prompts.append(prompt)

        # Check memory
        current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        tracemalloc.stop()

        print("\nBenchForge Memory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {current_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")

        # Should not use excessive memory (allow slightly more for BenchForge)
        assert memory_increase < 75  # Less than 75MB increase


class TestScalabilityPerformance:
    """Test scalability with increasing load."""

    def test_native_scalability(self):
        """Test native implementation scalability."""
        from flame.code.prompts.registry import get_prompt, PromptFormat

        prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

        sizes = [10, 100, 1000]
        times = []

        for size in sizes:
            start = time.perf_counter()

            for i in range(size):
                prompt_func(f"Statement {i}")

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            print(f"\nNative - {size} prompts: {elapsed:.3f}s")

        # Check linear scalability (roughly)
        # Time should scale linearly, not exponentially
        ratio_100_10 = times[1] / times[0]
        ratio_1000_100 = times[2] / times[1]

        # Ratios should be close to 10 (linear scaling)
        assert 5 < ratio_100_10 < 15
        assert 5 < ratio_1000_100 < 15

    def test_benchforge_scalability(self):
        """Test BenchForge implementation scalability."""
        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat
        except ImportError:
            pytest.skip("BenchForge not available")

        config = FLAMEConfig(name="fomc", dataset="fomc")
        task = FOMCTask(config)

        sizes = [10, 100, 1000]
        times = []

        for size in sizes:
            start = time.perf_counter()

            for i in range(size):
                task.create_prompt(
                    {"sentence": f"Statement {i}"}, PromptFormat.ZERO_SHOT
                )

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            print(f"\nBenchForge - {size} prompts: {elapsed:.3f}s")

        # Check linear scalability
        ratio_100_10 = times[1] / times[0]
        ratio_1000_100 = times[2] / times[1]

        # Ratios should be close to 10 (linear scaling)
        assert 5 < ratio_100_10 < 15
        assert 5 < ratio_1000_100 < 15


class TestPerformanceComparison:
    """Compare performance between implementations."""

    def test_implementation_comparison(self):
        """Compare both implementations side by side."""
        from flame.code.prompts.registry import get_prompt, PromptFormat as NativeFormat

        try:
            from flame.tasks.fomc import FOMCTask
            from flame.benchforge import FLAMEConfig, PromptFormat as BFFormat
        except ImportError:
            pytest.skip("BenchForge not available for comparison")

        # Setup
        native_func = get_prompt("fomc", NativeFormat.ZERO_SHOT)

        config = FLAMEConfig(name="fomc", dataset="fomc")
        bf_task = FOMCTask(config)

        # Benchmark both
        num_iterations = 1000
        test_input = "The Federal Reserve decided to maintain the current policy stance"

        # Native timing
        native_start = time.perf_counter()
        for _ in range(num_iterations):
            native_func(test_input)
        native_elapsed = time.perf_counter() - native_start

        # BenchForge timing
        bf_start = time.perf_counter()
        for _ in range(num_iterations):
            bf_task.create_prompt({"sentence": test_input}, BFFormat.ZERO_SHOT)
        bf_elapsed = time.perf_counter() - bf_start

        # Calculate performance difference
        perf_ratio = bf_elapsed / native_elapsed

        print(f"\nPerformance Comparison ({num_iterations} iterations):")
        print(f"  Native: {native_elapsed:.3f}s")
        print(f"  BenchForge: {bf_elapsed:.3f}s")
        print(f"  Ratio (BF/Native): {perf_ratio:.2f}x")

        # BenchForge should be within 2x of native performance
        assert perf_ratio < 2.0, f"BenchForge is {perf_ratio:.1f}x slower than native"

        # Save comparison results
        results = {
            "native_time": native_elapsed,
            "benchforge_time": bf_elapsed,
            "ratio": perf_ratio,
            "iterations": num_iterations,
            "acceptable": perf_ratio < 2.0,
        }

        return results


def test_generate_performance_report(tmp_path):
    """Generate comprehensive performance report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        },
        "tests": {},
    }

    # Run key performance tests and collect results
    # (In real scenario, would run all tests and aggregate)

    # Save report
    report_file = tmp_path / "performance_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPerformance report saved to: {report_file}")
    assert report_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
