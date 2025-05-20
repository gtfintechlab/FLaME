# Utils Directory Cleanup Analysis

This document contains the analysis of potentially unused or obsolete files and functions in the `src/flame/utils/` directory.

## Files That Appear Safe to Delete (No Observable Usage)

The following files appear to have no imports or references anywhere in the codebase:

1. **document_utils.py**: 
   - Purpose: Provides a function to split documents into chunks based on token count
   - Usage: No imports or references found
   - Dependencies: NLTK

2. **evaluate.py** (utils version): 
   - Purpose: Contains a class for evaluating summarization tasks using ChatGPT
   - Usage: No imports or references found
   - Dependencies: OpenAI, evaluate_ectsum

3. **evaluate_ectsum.py**: 
   - Purpose: Provides evaluation metrics for ECTSum (earnings call transcript summarization)
   - Usage: No imports or references found
   - Dependencies: ROUGE, BERTScore, formatter_ectsum

4. **evaluate_metrics.py**: 
   - Purpose: Similar to evaluate_ectsum but with a different class structure
   - Usage: No imports or references found
   - Dependencies: ROUGE, BERTScore

5. **formatter_ectsum.py**: 
   - Purpose: Formats dataframes for the ECTSum evaluation
   - Usage: No imports or references found
   - Dependencies: None

6. **fpb_data.py**: 
   - Purpose: Processing Financial Phrase Bank data
   - Usage: Contains imports from label_utils but isn't imported elsewhere
   - Dependencies: label_utils, logging_utils

7. **instructions.py**: 
   - Purpose: Contains configuration parameters
   - Usage: No direct imports found
   - Dependencies: None

8. **parse_test_bank.py**: 
   - Purpose: Parses test bank DOCX files into structured data
   - Usage: No imports or references found
   - Dependencies: python-docx

9. **prompt_generator.py**: 
   - Purpose: Functions to generate task-specific prompts
   - Usage: No imports or references found
   - Dependencies: None

10. **sampling_utils.py**: 
    - Purpose: Utility for sampling datasets
    - Usage: No imports or references found
    - Dependencies: datasets library

11. **test_utils.py**: 
    - Purpose: Provides a function for getting test output paths
    - Usage: No imports or references found
    - Dependencies: config.py

12. **word_count.py**: 
    - Purpose: Analyzes word counts in FOMC dataset
    - Usage: No imports or references found
    - Dependencies: NLTK, pandas, datasets

13. **datasets/dataset_utils.py**: 
    - Purpose: Provides encode/decode functions for datasets
    - Usage: No imports or references found
    - Dependencies: None

14. **results/** directory files: 
    - Purpose: Various results processing modules
    - Usage: No imports outside their own directory
    - Dependencies: Internal to results directory

## Files with Limited Usage (Keep These)

The following files have observable usage in the codebase and should be kept:

1. **logging_utils.py**: 
   - Purpose: Setup logging configuration
   - Usage: Widely used across the codebase
   - Primary function: `setup_logger`

2. **batch_utils.py**: 
   - Purpose: Batch processing functionality
   - Usage: Used in multiple inference/evaluation modules

3. **LabelMapper.py**: 
   - Purpose: Maps labels between formats
   - Usage: Used in NumClaim

4. **process_qa.py**: 
   - Purpose: Processing for QA tasks
   - Usage: Used in FinQA modules

5. **miscellaneous.py**: 
   - Purpose: Miscellaneous helper functions
   - Usage: Used in a FinQA module

6. **zip_to_csv.py**: 
   - Purpose: Converting zip files to CSV
   - Usage: Used in FinQA modules

## Recommended Approach for Cleanup

Before deleting any files:
1. Create a backup or work in a separate branch
2. Remove files one by one, running tests after each removal
3. Verify core functionality still works after all removals

This approach will help safely clean up the codebase while ensuring no critical functionality is lost.