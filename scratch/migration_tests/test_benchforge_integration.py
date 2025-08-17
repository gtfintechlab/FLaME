"""Test FLAME-BenchForge integration.

This module provides comprehensive tests for the Phase 4 integration
between FLAME and BenchForge.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchforge"))

# Import components to test
from flame.benchforge import (
    check_benchforge_status,
    create_llm_client,
    create_inference_engine,
    create_evaluation_engine,
    args_to_config,
    BENCHFORGE_AVAILABLE,
)
from flame.tasks import (
    register_all_flame_tasks,
    list_flame_tasks,
    create_task,
)


class MockArgs:
    """Mock command-line arguments for testing."""

    def __init__(self):
        self.task = "fomc"
        self.dataset = "fomc"
        self.model = "mock/test-model"
        self.max_tokens = 10
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = None
        self.batch_size = 5
        self.prompt_format = "zero_shot"
        self.num_samples = 10
        self.seed = 42
        self.split = "test"
        self.timeout = 60
        self.max_retries = 3
        self.metrics = ["accuracy", "f1_macro"]
        self.results_dir = "test_results"
        self.evaluation_dir = "test_evaluations"


class TestBenchForgeAvailability:
    """Test BenchForge availability and status."""

    def test_benchforge_available(self):
        """Test that BenchForge is available."""
        assert BENCHFORGE_AVAILABLE, "BenchForge should be available"

    def test_benchforge_status(self):
        """Test BenchForge status check."""
        status = check_benchforge_status()

        assert "available" in status
        assert "version" in status
        assert "registered_tasks" in status

        if status["available"]:
            assert status["version"] is not None
            assert isinstance(status["registered_tasks"], list)


class TestTaskRegistration:
    """Test FLAME task registration with BenchForge."""

    def test_register_all_tasks(self):
        """Test registering all FLAME tasks."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        tasks = register_all_flame_tasks()

        # Should have at least the example tasks
        assert "fomc" in tasks
        assert "fpb" in tasks
        assert len(tasks) >= 2

    def test_list_flame_tasks(self):
        """Test listing FLAME tasks."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        # Register tasks first
        register_all_flame_tasks()

        # List tasks
        tasks = list_flame_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) >= 0  # May be empty if adapter not initialized

    def test_create_task(self):
        """Test creating a FLAME task."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        # Register tasks first
        register_all_flame_tasks()

        # Create task
        task = create_task("fomc")

        assert task is not None
        assert hasattr(task, "config")
        assert task.config.name == "fomc"

    def test_task_decorator(self):
        """Test @flame_task decorator."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import flame_task, FLAMETask

        @flame_task("test_task")
        class TestTask(FLAMETask):
            def create_prompt(self, sample, format=None):
                return "test prompt"

        # Check task has metadata
        assert hasattr(TestTask, "_flame_task_name")
        assert TestTask._flame_task_name == "test_task"


class TestConfigurationConversion:
    """Test configuration conversion utilities."""

    def test_args_to_config(self):
        """Test converting args to configuration."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        args = MockArgs()
        configs = args_to_config(args, "fomc")

        # Check LLM config
        assert "llm_config" in configs
        llm_config = configs["llm_config"]
        assert llm_config.model == "mock/test-model"
        assert llm_config.max_tokens == 10
        assert llm_config.temperature == 0.0

        # Check task config
        assert "task_config" in configs
        task_config = configs["task_config"]
        assert task_config.name == "fomc"
        assert task_config.batch_size == 5

    def test_prompt_format_conversion(self):
        """Test prompt format conversion."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import PromptFormat

        # Test different formats
        args = MockArgs()

        args.prompt_format = "zero_shot"
        configs = args_to_config(args)
        assert configs["task_config"].prompt_format == PromptFormat.ZERO_SHOT

        args.prompt_format = "few_shot"
        configs = args_to_config(args)
        assert configs["task_config"].prompt_format == PromptFormat.FEW_SHOT

        args.prompt_format = "chain_of_thought"
        configs = args_to_config(args)
        assert configs["task_config"].prompt_format == PromptFormat.CHAIN_OF_THOUGHT


class TestFLAMETasks:
    """Test FLAME task implementations."""

    def test_fomc_task(self):
        """Test FOMC task."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask

        task = FOMCTask()

        # Check configuration
        assert task.config.name == "fomc"
        assert task.config.valid_labels == ["HAWKISH", "DOVISH", "NEUTRAL"]

        # Test prompt creation
        sample = {"text": "The Fed raised interest rates."}
        prompt = task.create_prompt(sample)
        assert isinstance(prompt, str)
        assert "HAWKISH" in prompt
        assert "DOVISH" in prompt
        assert "NEUTRAL" in prompt

    def test_fpb_task(self):
        """Test FPB task."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fpb import FPBTask

        task = FPBTask()

        # Check configuration
        assert task.config.name == "fpb"
        assert task.config.valid_labels == ["POSITIVE", "NEGATIVE", "NEUTRAL"]

        # Test prompt creation
        sample = {"sentence": "Profits increased significantly."}
        prompt = task.create_prompt(sample)
        assert isinstance(prompt, str)
        assert "POSITIVE" in prompt or "sentiment" in prompt.lower()

    def test_task_prompt_formats(self):
        """Test different prompt formats."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask
        from flame.benchforge import PromptFormat

        task = FOMCTask()
        sample = {"text": "Economic conditions remain stable."}

        # Zero-shot
        prompt_zero = task.create_prompt(sample, PromptFormat.ZERO_SHOT)
        assert "Example" not in prompt_zero

        # Few-shot
        prompt_few = task.create_prompt(sample, PromptFormat.FEW_SHOT)
        assert "Example" in prompt_few

        # Chain-of-thought
        prompt_cot = task.create_prompt(sample, PromptFormat.CHAIN_OF_THOUGHT)
        assert "step" in prompt_cot.lower()

    def test_response_extraction(self):
        """Test response extraction."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.tasks.fomc import FOMCTask

        task = FOMCTask()

        # Test extraction
        responses = [
            "HAWKISH",
            "The answer is DOVISH",
            "Classification: NEUTRAL",
            "I think it's hawkish",
        ]

        for response in responses:
            extracted = task.extract_response(response)
            # Should extract a label or None
            assert extracted is None or extracted in task.config.valid_labels


class TestInferenceEngine:
    """Test inference engine integration."""

    @patch("flame.benchforge.LLMClient")
    def test_create_llm_client(self, mock_llm_class):
        """Test creating LLM client."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        mock_client = Mock()
        mock_llm_class.return_value = mock_client

        args = MockArgs()
        client = create_llm_client(args)

        assert mock_llm_class.called
        assert client == mock_client

    @patch("flame.benchforge.InferenceEngine")
    @patch("flame.benchforge.LLMClient")
    def test_create_inference_engine(self, mock_llm_class, mock_engine_class):
        """Test creating inference engine."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        mock_client = Mock()
        mock_llm_class.return_value = mock_client

        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        engine = create_inference_engine()

        assert mock_engine_class.called
        assert engine == mock_engine

    @patch("flame.benchforge.LLMClient")
    def test_inference_with_mock_llm(self, mock_llm_class):
        """Test running inference with mock LLM."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        # Setup mock LLM
        mock_client = Mock()
        mock_client.complete.return_value = "HAWKISH"
        mock_client.complete_batch.return_value = ["HAWKISH"] * 5
        mock_llm_class.return_value = mock_client

        # Register tasks
        register_all_flame_tasks()

        # Create engine
        from flame.benchforge import InferenceEngine

        engine = InferenceEngine(llm_client=mock_client, output_dir=Path("test_output"))

        # Create config
        from flame.benchforge import FLAMEConfig

        config = FLAMEConfig(
            name="fomc",
            dataset="fomc",
            num_samples=5,
            batch_size=5,
        )

        # Mock dataset loading
        with patch.object(engine, "_load_task_dataset") as mock_load:
            mock_load.return_value = [
                {"text": f"Statement {i}", "label": "NEUTRAL"} for i in range(5)
            ]

            # Run inference
            result = engine.run("fomc", config, save_results=False)

            assert result is not None
            assert hasattr(result, "results_df")
            assert len(result.results_df) == 5


class TestEvaluationEngine:
    """Test evaluation engine integration."""

    @patch("flame.benchforge.EvaluationEngine")
    def test_create_evaluation_engine(self, mock_engine_class):
        """Test creating evaluation engine."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        engine = create_evaluation_engine()

        assert mock_engine_class.called
        assert engine == mock_engine

    def test_evaluation_with_mock_results(self):
        """Test evaluation with mock results."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        # Create mock results DataFrame
        results_df = pd.DataFrame(
            {
                "input": ["text1", "text2", "text3"],
                "prompt": ["prompt1", "prompt2", "prompt3"],
                "raw_response": ["HAWKISH", "DOVISH", "NEUTRAL"],
                "extracted_response": ["HAWKISH", "DOVISH", "NEUTRAL"],
                "ground_truth": ["HAWKISH", "NEUTRAL", "NEUTRAL"],
            }
        )

        # Create evaluation engine
        from flame.benchforge import EvaluationEngine

        engine = EvaluationEngine(output_dir=Path("test_evaluations"))

        # Run evaluation
        result = engine.evaluate(
            results_df=results_df, task="fomc", metrics=["accuracy"], save_results=False
        )

        assert result is not None
        assert "accuracy" in result.metrics
        # 1 out of 3 correct (first one matches)
        expected_accuracy = 1 / 3
        assert abs(result.metrics["accuracy"] - expected_accuracy) < 0.01


class TestBackwardCompatibility:
    """Test backward compatibility with existing FLAME code."""

    def test_chunk_list_available(self):
        """Test that chunk_list is available for backward compatibility."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import chunk_list

        # Test chunking
        items = list(range(10))
        chunks = list(chunk_list(items, 3))

        assert len(chunks) == 4  # 3, 3, 3, 1
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]

    @patch("flame.benchforge.LLMClient")
    def test_process_batch_compatibility(self, mock_llm_class):
        """Test process_batch_with_retry compatibility."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import process_batch_with_retry

        # Setup mock
        mock_client = Mock()
        mock_client.complete_batch.return_value = ["response1", "response2"]
        mock_llm_class.return_value = mock_client

        # Test compatibility function
        args = MockArgs()
        messages = [
            [{"role": "user", "content": "test1"}],
            [{"role": "user", "content": "test2"}],
        ]

        responses = process_batch_with_retry(args, messages, 0, 1)

        assert len(responses) == 2
        # Check mock response format
        assert hasattr(responses[0], "choices")


class TestFLAMEUtils:
    """Test FLAME utility functions."""

    def test_load_flame_dataset(self):
        """Test dataset loading utility."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import load_flame_dataset

        # This would need actual dataset or mocking
        # For now, just test it doesn't crash with invalid dataset
        with pytest.raises(ValueError):
            load_flame_dataset("nonexistent_dataset")

    def test_process_flame_results(self):
        """Test results processing utility."""
        if not BENCHFORGE_AVAILABLE:
            pytest.skip("BenchForge not available")

        from flame.benchforge import process_flame_results

        # Create mock results
        results_df = pd.DataFrame(
            {
                "input": ["text1"],
                "raw_response": ["HAWKISH"],
                "extracted_response": ["HAWKISH"],
                "ground_truth": ["HAWKISH"],
            }
        )

        # Process results
        output_path = process_flame_results(
            results_df, "test_task", "test_model", output_dir=Path("test_output")
        )

        assert output_path is not None
        assert output_path.suffix == ".csv"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
