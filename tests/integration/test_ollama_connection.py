#!/usr/bin/env python3
"""Test script to verify Ollama integration with LiteLLM."""

from litellm import completion


def test_ollama_connection():
    """Test basic Ollama connection through LiteLLM."""
    print("Testing Ollama connection through LiteLLM...")

    try:
        # Test 1: Basic completion
        print("\n1. Testing basic completion...")
        response = completion(
            model="ollama/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_base="http://localhost:11434",
            temperature=0.0,
            max_tokens=50,
        )
        print("✓ Basic completion successful!")
        print(f"Response: {response.choices[0].message.content}")

        # Test 2: Financial reasoning (similar to FLaME tasks)
        print("\n2. Testing financial reasoning...")
        financial_prompt = """Calculate the profit margin if revenue is $1000 and costs are $750.

        Please provide the answer as a percentage."""

        response = completion(
            model="ollama/qwen2.5:1.5b",
            messages=[{"role": "user", "content": financial_prompt}],
            api_base="http://localhost:11434",
            temperature=0.0,
            max_tokens=100,
        )
        print("✓ Financial reasoning successful!")
        print(f"Response: {response.choices[0].message.content}")

        # Test 3: Check response format compatibility
        print("\n3. Checking response format...")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Choices: {len(response.choices)}")

        print("\n✅ All tests passed! Ollama is working correctly with LiteLLM.")
        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: 'ollama serve'")
        print("2. Check if qwen2.5:1.5b is installed: 'ollama list'")
        print("3. Pull the model if needed: 'ollama pull qwen2.5:1.5b'")
        return False


def test_batch_completion():
    """Test batch completion capabilities."""
    print("\n4. Testing batch completion...")
    try:
        from litellm import batch_completion

        messages = [
            [{"role": "user", "content": "What is 10% of 100?"}],
            [{"role": "user", "content": "What is 25% of 80?"}],
        ]

        responses = batch_completion(
            model="ollama/qwen2.5:1.5b",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=0.0,
            max_tokens=50,
        )

        print("✓ Batch completion successful!")
        for i, response in enumerate(responses):
            print(f"Response {i + 1}: {response.choices[0].message.content}")

        return True
    except Exception as e:
        print(f"❌ Batch completion error: {str(e)}")
        return False


if __name__ == "__main__":
    print("Ollama Integration Test for FLaME")
    print("=" * 50)

    # Test basic connection
    if test_ollama_connection():
        # Test batch completion
        test_batch_completion()

    print("\nNote: To use Ollama in FLaME, run:")
    print(
        "uv run python main.py --config configs/development.yaml --mode inference --dataset fomc"
    )
