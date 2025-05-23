"""Test lazy loading of BERTScore in evaluation modules."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
def test_ectsum_bertscore_lazy_loading():
    """Test that ectsum loads BERTScore lazily and handles errors properly."""

    # Import the module - this should NOT trigger BERTScore loading
    from flame.code.ectsum.ectsum_evaluate import (
        get_bertscore,
        _bertscore,
        _bertscore_error,
    )

    # Verify BERTScore is not loaded yet
    assert _bertscore is None
    assert _bertscore_error is None

    # Test successful loading
    with patch("flame.code.ectsum.ectsum_evaluate.load") as mock_load:
        mock_bertscore = MagicMock()
        mock_load.return_value = mock_bertscore

        # First call should load BERTScore
        result = get_bertscore()
        assert result == mock_bertscore
        mock_load.assert_called_once_with("bertscore")

        # Second call should return cached instance
        result2 = get_bertscore()
        assert result2 == mock_bertscore
        mock_load.assert_called_once()  # Still only called once


@pytest.mark.unit
def test_ectsum_bertscore_loading_error():
    """Test that ectsum handles BERTScore loading errors properly."""

    # Reset the module state
    import importlib
    import flame.code.ectsum.ectsum_evaluate

    importlib.reload(flame.code.ectsum.ectsum_evaluate)

    from flame.code.ectsum.ectsum_evaluate import get_bertscore

    # Test error handling
    with patch("flame.code.ectsum.ectsum_evaluate.load") as mock_load:
        mock_load.side_effect = ImportError("No module named 'bert_score'")

        # First call should raise error
        with pytest.raises(RuntimeError) as exc_info:
            get_bertscore()
        assert "Failed to load BERTScore metric" in str(exc_info.value)
        assert "bert-score" in str(exc_info.value)

        # Second call should raise the same cached error without trying to load again
        with pytest.raises(RuntimeError) as exc_info2:
            get_bertscore()
        assert str(exc_info.value) == str(exc_info2.value)
        mock_load.assert_called_once()  # Only tried once


@pytest.mark.unit
def test_edtsum_bertscore_lazy_loading():
    """Test that edtsum loads BERTScore lazily and handles errors properly."""

    # Reset the module state
    import importlib
    import flame.code.edtsum.edtsum_evaluate

    importlib.reload(flame.code.edtsum.edtsum_evaluate)

    from flame.code.edtsum.edtsum_evaluate import get_bertscore

    # Test successful loading
    with patch("flame.code.edtsum.edtsum_evaluate.load") as mock_load:
        mock_bertscore = MagicMock()
        mock_load.return_value = mock_bertscore

        result = get_bertscore()
        assert result == mock_bertscore
        mock_load.assert_called_once_with("bertscore")


@pytest.mark.unit
def test_importing_modules_without_bertscore():
    """Test that we can import evaluation modules without BERTScore installed."""

    # This should work without errors even if bert-score is not installed
    import flame.code.ectsum.ectsum_evaluate
    import flame.code.edtsum.edtsum_evaluate

    # The modules should be importable
    assert hasattr(flame.code.ectsum.ectsum_evaluate, "get_bertscore")
    assert hasattr(flame.code.edtsum.edtsum_evaluate, "get_bertscore")
