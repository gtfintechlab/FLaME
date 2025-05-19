#!/usr/bin/env python3
"""Unit tests for dataset sampling logic"""

import pytest
from datasets import Dataset
from unittest.mock import Mock, patch


class TestDatasetSampling:
    """Test the dataset sampling logic isolated from inference"""
    
    def test_sample_selection_logic(self):
        """Test the logic for selecting samples from dataset"""
        # Create a mock dataset
        data = [{"text": f"sample {i}", "label": i} for i in range(100)]
        dataset = Dataset.from_list(data)
        
        # Test different sample sizes
        test_cases = [
            (10, 10),    # Request 10, get 10
            (50, 50),    # Request 50, get 50
            (100, 100),  # Request 100, get all 100
            (150, 100),  # Request 150, get only 100 (dataset size)
            (0, 0),      # Request 0, get 0
        ]
        
        for requested, expected in test_cases:
            subset = dataset.select(range(min(requested, len(dataset))))
            assert len(subset) == expected, f"Failed for requested={requested}"
    
    def test_fpb_dataset_structure(self):
        """Test that FPB dataset has expected structure"""
        # Mock the dataset structure
        fpb_data = []
        for i in range(20):
            fpb_data.append({
                "sentence": f"Financial news sentence {i}",
                "label": i % 3  # 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE
            })
        
        dataset = Dataset.from_list(fpb_data)
        
        # Verify structure
        assert "sentence" in dataset.column_names
        assert "label" in dataset.column_names
        assert len(dataset) == 20
        
        # Test selecting subset
        subset = dataset.select(range(5))
        assert len(subset) == 5
        assert subset[0]["sentence"] == "Financial news sentence 0"
        assert subset[4]["sentence"] == "Financial news sentence 4"
    
    def test_fomc_dataset_structure(self):
        """Test that FOMC dataset has expected structure"""
        # Mock the dataset structure
        fomc_data = []
        labels = ["DOVISH", "HAWKISH", "NEUTRAL"]
        for i in range(15):
            fomc_data.append({
                "sentence": f"FOMC statement {i}",
                "label": labels[i % 3]
            })
        
        dataset = Dataset.from_list(fomc_data)
        
        # Verify structure
        assert "sentence" in dataset.column_names
        assert "label" in dataset.column_names
        assert len(dataset) == 15
        
        # Test selecting subset
        subset = dataset.select(range(10))
        assert len(subset) == 10
        assert subset[0]["label"] in ["DOVISH", "HAWKISH", "NEUTRAL"]
    
    def test_sample_size_none_behavior(self):
        """Test behavior when sample_size is None (use all data)"""
        data = [{"text": f"sample {i}"} for i in range(50)]
        dataset = Dataset.from_list(data)
        
        # Simulate the logic in inference functions
        args = Mock()
        args.sample_size = None
        
        if hasattr(args, 'sample_size') and args.sample_size is not None:
            subset = dataset.select(range(min(args.sample_size, len(dataset))))
        else:
            subset = dataset  # Use all data
        
        assert len(subset) == 50  # Should use all data
    
    def test_sample_size_attribute_missing(self):
        """Test behavior when args doesn't have sample_size attribute"""
        data = [{"text": f"sample {i}"} for i in range(30)]
        dataset = Dataset.from_list(data)
        
        # Args without sample_size attribute
        args = Mock(spec=[])  # Empty spec means no attributes
        
        if hasattr(args, 'sample_size') and args.sample_size is not None:
            subset = dataset.select(range(min(args.sample_size, len(dataset))))
        else:
            subset = dataset  # Use all data
        
        assert len(subset) == 30  # Should use all data
    
    def test_batch_processing_with_samples(self):
        """Test that batch processing works correctly with limited samples"""
        from flame.utils.batch_utils import chunk_list
        
        # Create sample data
        sentences = [f"sentence {i}" for i in range(23)]
        
        # Test different batch sizes
        batch_size = 5
        batches = chunk_list(sentences, batch_size)
        
        assert len(batches) == 5  # 23/5 = 4.6, so 5 batches
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert len(batches[2]) == 5
        assert len(batches[3]) == 5
        assert len(batches[4]) == 3  # Last batch has remainder
        
        # Verify content
        assert batches[0][0] == "sentence 0"
        assert batches[4][2] == "sentence 22"
    
    def test_sample_indices_are_sequential(self):
        """Test that samples are taken sequentially from start"""
        data = [{"id": i, "text": f"sample {i}"} for i in range(100)]
        dataset = Dataset.from_list(data)
        
        # Select first 10 samples
        subset = dataset.select(range(10))
        
        # Verify they are the first 10 items
        for i in range(10):
            assert subset[i]["id"] == i
            assert subset[i]["text"] == f"sample {i}"
    
    @patch('datasets.load_dataset')
    def test_real_dataset_loading_mock(self, mock_load_dataset):
        """Test with mocked real dataset loading"""
        # Mock the FPB dataset
        mock_data = []
        for i in range(200):
            mock_data.append({
                "sentence": f"The company reported strong earnings in Q{i%4+1}",
                "label": i % 3
            })
        
        mock_dataset = {"test": Dataset.from_list(mock_data)}
        mock_load_dataset.return_value = mock_dataset
        
        # Load dataset
        dataset = mock_load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree")
        test_data = dataset["test"]
        
        # Apply sample limit
        sample_size = 25
        limited_data = test_data.select(range(min(sample_size, len(test_data))))
        
        assert len(limited_data) == 25
        assert limited_data[0]["sentence"].startswith("The company reported")


if __name__ == "__main__":
    # Run a simple test
    test = TestDatasetSampling()
    test.test_sample_selection_logic()
    print("Dataset sampling tests passed!")