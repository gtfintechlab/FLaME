# Project TODOs

*Generated on 2024-12-14 14:35:18*

## superflue/code/mmlu/mmlu_loader.py

- Line 10: remove the class stuff from mmlu, and re-ennable logging

## superflue/utils/LabelMapper.py

- Line 1: (Glenn) need to move LabelMapper into another file, no single-function files

## superflue/utils/batch_utils.py

- Line 85: This function does not currently have a hard guarantee it will use the right model for extraction/inference. If an extraction model is passed, it will use the inference model for the extraction. This is a major problem!!!

## superflue/utils/document_utils.py

- Line 1: (Glenn) Evaluate code will need to be moved into its own folder not utils.
- Line 3: (Glenn) I prefer to avoid using NLTK unless there's something we cannot get from another package

## superflue/utils/evaluate.py

- Line 3: (Glenn) Evaluate code will need to be moved into its own folder not utils.

## superflue/utils/evaluate_ectsum.py

- Line 1: (Glenn) Evaluate code will need to be moved into its own folder not utils.

## superflue/utils/evaluate_metrics.py

- Line 3: (Glenn) Evaluate code will need to be moved into its own folder not utils.

## superflue/utils/formatter_ectsum.py

- Line 1: (Glenn) If this is a formatter just for ectsum we should keep it in the associated data\task folder not here

## superflue/utils/fpb_data.py

- Line 35: have it build to the directories when reused i.e. numclaim_detection

## superflue/utils/label_utils.py

- Line 1: (Glenn) superflue.utils.label_utils is one function, it doesnt need to exist

## superflue/utils/miscellaneous.py

- Line 5: (Glenn) superflue.utils.miscellaneous is one function -- it can be refactored elsewhere and deleted

## superflue/utils/results/decode.py

- Line 2: (Glenn) Lets refactor all the decode/encode functions together into one place

## superflue/utils/results/finer_ord_generative_llm_res.py

- Line 1: (Glenn) I do not think we are using this file in SuperFLUE so it can be deleted/moved -- please double check before deleting!!

## superflue/utils/results/fomc_communication_generative_llm_res.py

- Line 1: (Glenn) I do not think we are using this file in SuperFLUE so it can be deleted/moved -- please double check before deleting!!

## superflue/utils/results/numclaim_detection_generative_llm_res.py

- Line 1: (Glenn) I do not think we are using this file in SuperFLUE so it can be deleted/moved -- please double check before deleting!!

## superflue/utils/sampling_utils.py

- Line 7: (Glenn) Adapt this function to be the standard dataset loader for all SuperFLUE tasks.

## superflue/utils/word_count.py

- Line 4: (Glenn) If possible, avoid using `nltk`, unless we need something specific not found elsewhere
- Line 12: (Glenn) `word_count.py` is written very strictly and should be either a utility we call on a dataset we provide, or something done at runtime, etc.

## superflue/utils/zip_to_csv.py

- Line 6: (Glenn) We really want to avoid one-function-one-file pattern

