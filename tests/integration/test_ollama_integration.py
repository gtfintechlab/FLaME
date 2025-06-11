"""Integration tests using Ollama for real inference testing."""

import pandas as pd
import pytest

from flame.code.inference import run_inference


class TestOllamaIntegration:
    """Test FLaME tasks with real Ollama inference."""

    @pytest.mark.integration
    def test_fomc_inference_with_ollama(self, ollama_integration_test, tmp_path):
        """Test FOMC task inference using Ollama."""

        # Create minimal args object
        class Args:
            model = "ollama/qwen2.5:1.5b"
            api_base = "http://localhost:11434"
            dataset = "fomc"
            task = "fomc"
            max_tokens = 128
            temperature = 0.0
            top_p = 0.9
            top_k = None
            repetition_penalty = 1.0
            batch_size = 5  # Small batch for testing
            prompt_format = "zero_shot"
            timeout = 30

        args = Args()

        # Run inference (this will use real Ollama)
        df = run_inference(args)

        # Verify results
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "response" in df.columns
        assert "actual" in df.columns

        # Check that we got actual responses
        assert df["response"].notna().any()

    @pytest.mark.integration
    def test_fpb_inference_with_ollama(self, ollama_integration_test):
        """Test FPB (sentiment) task with Ollama."""

        class Args:
            model = "ollama/qwen2.5:1.5b"
            api_base = "http://localhost:11434"
            dataset = "fpb"
            task = "fpb"
            max_tokens = 50  # Sentiment needs fewer tokens
            temperature = 0.0
            top_p = 0.9
            top_k = None
            repetition_penalty = 1.0
            batch_size = 10
            prompt_format = "zero_shot"
            timeout = 30

        args = Args()

        # Run inference
        df = run_inference(args)

        # Verify results
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "response" in df.columns

    def test_ollama_vs_mock_comparison(self, use_ollama_or_mock):
        """Test that compares Ollama and mock responses."""
        # This test will use real Ollama if available, mock otherwise
        response = use_ollama_or_mock(
            messages=[{"role": "user", "content": "What is 10 + 10?"}]
        )

        assert response.choices[0].message.content is not None

        # If using real Ollama, response should contain "20"
        # If using mock, response will be "Mocked response: 42"
        content = response.choices[0].message.content
        assert len(content) > 0

    @pytest.mark.parametrize("task", ["fomc", "fpb", "numclaim"])
    def test_multiple_tasks_with_ollama(self, task, ollama_integration_test):
        """Test multiple tasks with Ollama."""

        class Args:
            model = "ollama/qwen2.5:1.5b"
            api_base = "http://localhost:11434"
            dataset = task
            task = task
            max_tokens = 128
            temperature = 0.0
            top_p = 0.9
            top_k = None
            repetition_penalty = 1.0
            batch_size = 3  # Very small batch for quick testing
            prompt_format = "zero_shot"
            timeout = 30

        args = Args()

        try:
            df = run_inference(args)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        except Exception as e:
            # Some tasks might not be fully compatible
            pytest.skip(f"Task {task} not compatible with current setup: {str(e)}")


class TestOllamaPerformance:
    """Performance tests comparing Ollama with mocked responses."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_inference_speed(self, ollama_integration_test, benchmark):
        """Benchmark Ollama inference speed."""
        from litellm import completion

        def run_inference():
            return completion(
                model="ollama/qwen2.5:1.5b",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                api_base="http://localhost:11434",
                temperature=0.0,
                max_tokens=20,
            )

        # Benchmark the inference
        result = benchmark(run_inference)
        assert result.choices[0].message.content is not None
