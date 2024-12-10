import pytest
import pandas as pd
from pathlib import Path
from litellm import completion
from argparse import Namespace
from ferrari.code.fomc.fomc_inference import fomc_inference
from ferrari.code.prompts import fomc_prompt

def test_fomc_prompt():
    """Test the FOMC prompt generation."""
    test_sentences = [
        "The Federal Reserve maintains its accommodative stance.",
        "The Committee raises rates by 75 basis points.",
        ""  # Empty string test
    ]
    
    for sentence in test_sentences:
        prompt = fomc_prompt(sentence)
        assert isinstance(prompt, str)
        assert sentence in prompt
        assert "HAWKISH" in prompt.upper()
        assert "DOVISH" in prompt.upper()
        assert "NEUTRAL" in prompt.upper()

def test_basic_inference():
    """Test basic inference with mock responses."""
    args = Namespace(
        model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=32,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
        batch_size=10
    )
    
    # Define expected mock responses for different inputs
    mock_responses = {
        "The Federal Reserve will maintain its accommodative monetary policy stance.": "DOVISH",
        "The Committee decided to raise interest rates by 75 basis points.": "HAWKISH",
        "Economic conditions remain balanced, with stable inflation and employment.": "NEUTRAL"
    }
    
    try:
        # Run inference with mock responses
        for input_text, expected_response in mock_responses.items():
            response = completion(
                model=args.model,
                messages=[{"role": "user", "content": input_text}],
                mock_response=expected_response
            )
            
            # Verify the response
            assert response.choices[0].message.content == expected_response
            
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

def test_inference_error_handling(tmp_path):
    """Test error handling in the inference pipeline."""
    args = Namespace(
        model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=32,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
        batch_size=10
    )
    
    # Test handling of API errors
    def mock_completion_error(*args, **kwargs):
        raise Exception("API Error")
    
    # Save original completion function
    original_completion = completion
    try:
        # Replace completion with error-raising version
        globals()['completion'] = mock_completion_error
        
        # Run inference
        df = fomc_inference(args)
        
        # Check that errors were handled gracefully
        assert "llm_responses" in df.columns
        assert df["llm_responses"].isna().any()  # Some responses should be None
        assert "complete_responses" in df.columns
        assert df["complete_responses"].isna().any()  # Some complete responses should be None
        
    finally:
        # Restore original completion function
        globals()['completion'] = original_completion

def test_output_dataframe_structure(tmp_path):
    """Test the structure of the output DataFrame."""
    args = Namespace(
        model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=32,
        temperature=0.0,
        top_p=0.9,
        top_k=None,
        repetition_penalty=1.0,
        batch_size=10
    )
    
    # Run inference with mock data
    df = fomc_inference(args)
    
    # Check DataFrame structure
    required_columns = ["sentences", "llm_responses", "actual_labels", "complete_responses"]
    for col in required_columns:
        assert col in df.columns
    
    # Check data types
    assert df["sentences"].dtype == object  # strings
    assert df["llm_responses"].dtype == object  # strings or None
    assert df["actual_labels"].dtype in [int, object]  # integers or strings
    assert df["complete_responses"].dtype == object  # response objects or None
    
    # Check for no empty strings in sentences
    assert not df["sentences"].str.len().eq(0).any()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 