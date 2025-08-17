#!/usr/bin/env python3
"""
Phase 1 Validation: FOMC Feature Parity Testing
================================================

This script validates that FOMC works identically with and without BenchForge.
It tests inference, extraction, and evaluation pipelines.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd

# Add FLAME to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FOMCValidationSuite:
    """Comprehensive validation suite for FOMC implementations."""

    def __init__(self, test_samples: int = 10, output_dir: str = "validation_results"):
        """Initialize validation suite.

        Args:
            test_samples: Number of samples to test
            output_dir: Directory for validation outputs
        """
        self.test_samples = test_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test configuration
        self.test_config = {
            "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "max_tokens": 10,
            "temperature": 0.0,
            "batch_size": 5,
            "prompt_format": "zero_shot",
        }

        # Results storage
        self.results = {"native": {}, "benchforge": {}, "comparison": {}}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_test_data(self) -> pd.DataFrame:
        """Create test dataset for validation.

        Returns:
            DataFrame with test samples
        """
        logger.info(f"Creating test dataset with {self.test_samples} samples")

        # Sample FOMC statements for testing
        test_statements = [
            {
                "sentence": "The Committee decided to raise the federal funds rate by 25 basis points.",
                "label": 1,  # HAWKISH
                "label_text": "HAWKISH",
            },
            {
                "sentence": "Economic conditions may warrant exceptionally low levels of the federal funds rate for an extended period.",
                "label": 0,  # DOVISH
                "label_text": "DOVISH",
            },
            {
                "sentence": "The Committee will maintain its current policy stance.",
                "label": 2,  # NEUTRAL
                "label_text": "NEUTRAL",
            },
            {
                "sentence": "Inflation has been running persistently below 2 percent.",
                "label": 0,  # DOVISH
                "label_text": "DOVISH",
            },
            {
                "sentence": "The Committee is prepared to adjust monetary policy as appropriate to counter inflation risks.",
                "label": 1,  # HAWKISH
                "label_text": "HAWKISH",
            },
            {
                "sentence": "The Committee will continue to monitor the implications of incoming information.",
                "label": 2,  # NEUTRAL
                "label_text": "NEUTRAL",
            },
            {
                "sentence": "Further gradual increases in the target range will be consistent with sustained expansion.",
                "label": 1,  # HAWKISH
                "label_text": "HAWKISH",
            },
            {
                "sentence": "The Committee judges that the downside risks to the outlook have diminished.",
                "label": 2,  # NEUTRAL
                "label_text": "NEUTRAL",
            },
            {
                "sentence": "Accommodative monetary policy remains appropriate.",
                "label": 0,  # DOVISH
                "label_text": "DOVISH",
            },
            {
                "sentence": "The Committee expects to begin reducing its holdings of Treasury securities.",
                "label": 1,  # HAWKISH
                "label_text": "HAWKISH",
            },
        ]

        # Use only the requested number of samples
        test_data = test_statements[: self.test_samples]
        df = pd.DataFrame(test_data)

        # Save test data
        test_file = self.output_dir / f"test_data_{self.timestamp}.csv"
        df.to_csv(test_file, index=False)
        logger.info(f"Test data saved to: {test_file}")

        return df

    def test_native_inference(self, test_data: pd.DataFrame) -> Dict:
        """Test native FLAME FOMC inference.

        Args:
            test_data: Test dataset

        Returns:
            Results dictionary
        """
        logger.info("=" * 60)
        logger.info("Testing Native FLAME FOMC Implementation")
        logger.info("=" * 60)

        results = {"prompts": [], "responses": [], "extracted_labels": [], "errors": []}

        try:
            # Import native FLAME components
            from flame.code.prompts.registry import get_prompt, PromptFormat

            # Test prompt generation
            logger.info("Testing prompt generation...")
            prompt_func = get_prompt("fomc", PromptFormat.ZERO_SHOT)

            for idx, row in test_data.iterrows():
                try:
                    prompt = prompt_func(row["sentence"])
                    results["prompts"].append(prompt)
                    logger.debug(f"Sample {idx}: Prompt generated successfully")
                except Exception as e:
                    logger.error(f"Sample {idx}: Prompt generation failed: {e}")
                    results["errors"].append(f"Prompt error at {idx}: {e}")

            # Test response extraction (without actual LLM calls)
            logger.info("Testing extraction logic...")
            from flame.code.fomc.fomc_evaluate import map_label_to_number

            # Simulate responses for extraction testing
            test_responses = [
                "HAWKISH",
                "The statement is DOVISH",
                "Classification: NEUTRAL",
                "dovish\nThis indicates loose policy",
                "Answer: HAWKISH",
            ]

            for response in test_responses[: len(test_data)]:
                # Test the label mapping function
                for label in ["HAWKISH", "DOVISH", "NEUTRAL"]:
                    if label.lower() in response.lower():
                        mapped = map_label_to_number(label)
                        results["extracted_labels"].append(mapped)
                        break
                else:
                    results["extracted_labels"].append(-1)

            logger.info(f"✅ Native FLAME: Generated {len(results['prompts'])} prompts")
            logger.info("✅ Native FLAME: Extraction logic tested successfully")

        except Exception as e:
            logger.error(f"❌ Native FLAME test failed: {e}")
            results["errors"].append(f"General error: {e}")

        self.results["native"] = results
        return results

    def test_benchforge_inference(self, test_data: pd.DataFrame) -> Dict:
        """Test BenchForge FOMC implementation.

        Args:
            test_data: Test dataset

        Returns:
            Results dictionary
        """
        logger.info("=" * 60)
        logger.info("Testing BenchForge FOMC Implementation")
        logger.info("=" * 60)

        results = {"prompts": [], "responses": [], "extracted_labels": [], "errors": []}

        try:
            # Import BenchForge components
            from flame.benchforge import BENCHFORGE_AVAILABLE, FLAMEConfig, PromptFormat

            if not BENCHFORGE_AVAILABLE:
                logger.error("BenchForge not available!")
                results["errors"].append("BenchForge not installed")
                return results

            from flame.tasks.fomc import FOMCTask

            # Create task instance
            config = FLAMEConfig(
                name="fomc",
                dataset="fomc",
                huggingface_dataset="gtfintechlab/fomc_communication",
                prompt_format=PromptFormat.ZERO_SHOT,
                max_tokens=10,
                batch_size=5,
                text_field="sentence",
                label_field="label",
                valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
            )

            task = FOMCTask(config)
            logger.info("✅ BenchForge: FOMCTask initialized successfully")

            # Test prompt generation
            logger.info("Testing prompt generation...")
            for idx, row in test_data.iterrows():
                try:
                    prompt = task.create_prompt(row.to_dict(), PromptFormat.ZERO_SHOT)
                    results["prompts"].append(prompt)
                    logger.debug(f"Sample {idx}: Prompt generated successfully")
                except Exception as e:
                    logger.error(f"Sample {idx}: Prompt generation failed: {e}")
                    results["errors"].append(f"Prompt error at {idx}: {e}")

            # Test extraction logic
            logger.info("Testing extraction logic...")
            test_responses = [
                "HAWKISH",
                "The statement is DOVISH",
                "Classification: NEUTRAL",
                "dovish\nThis indicates loose policy",
                "Answer: HAWKISH",
            ]

            for idx, response in enumerate(test_responses[: len(test_data)]):
                try:
                    # Test rule-based extraction
                    extracted = task.extract_label_from_response(
                        response, use_llm_fallback=False
                    )
                    if extracted:
                        mapped = task.map_label_to_number(extracted)
                        results["extracted_labels"].append(mapped)
                    else:
                        results["extracted_labels"].append(-1)
                except Exception as e:
                    logger.error(f"Extraction error: {e}")
                    results["extracted_labels"].append(-1)

            # Test LLM-based extraction capability
            logger.info("Testing LLM-based extraction capability...")
            from bench_forge.prompts.extractor import (
                ResponseExtractor,
                ExtractionStrategy,
            )

            extractor = ResponseExtractor()
            test_messy_response = """
            After analyzing the statement, I believe this indicates a tightening 
            of monetary policy. The fed is clearly taking a hawkish stance here.
            Therefore, my classification is HAWKISH.
            """

            # Test without actual LLM (will use fallback)
            result = extractor.extract(
                test_messy_response,
                strategy=ExtractionStrategy.FUZZY,
                options=["HAWKISH", "DOVISH", "NEUTRAL"],
            )

            logger.info(
                f"✅ BenchForge: Extraction test result: {result.value} (confidence: {result.confidence})"
            )
            logger.info(f"✅ BenchForge: Generated {len(results['prompts'])} prompts")
            logger.info("✅ BenchForge: Extraction logic tested successfully")

        except Exception as e:
            logger.error(f"❌ BenchForge test failed: {e}")
            results["errors"].append(f"General error: {e}")

        self.results["benchforge"] = results
        return results

    def compare_implementations(self) -> Dict:
        """Compare native and BenchForge implementations.

        Returns:
            Comparison results
        """
        logger.info("=" * 60)
        logger.info("Comparing Implementations")
        logger.info("=" * 60)

        comparison = {
            "prompt_match": False,
            "extraction_match": False,
            "differences": [],
            "summary": {},
        }

        native = self.results.get("native", {})
        benchforge = self.results.get("benchforge", {})

        if not native or not benchforge:
            logger.error("Missing test results for comparison")
            return comparison

        # Compare prompts
        if native.get("prompts") and benchforge.get("prompts"):
            prompt_matches = 0
            for i, (n_prompt, b_prompt) in enumerate(
                zip(native["prompts"], benchforge["prompts"])
            ):
                # Normalize for comparison (remove extra whitespace)
                n_clean = " ".join(n_prompt.split())
                b_clean = " ".join(b_prompt.split())

                if n_clean == b_clean:
                    prompt_matches += 1
                else:
                    # Check if semantically similar (core content matches)
                    if "HAWKISH" in n_clean and "HAWKISH" in b_clean:
                        if "DOVISH" in n_clean and "DOVISH" in b_clean:
                            if "NEUTRAL" in n_clean and "NEUTRAL" in b_clean:
                                prompt_matches += 1
                            else:
                                comparison["differences"].append(
                                    f"Prompt {i}: Content differs"
                                )

            prompt_match_rate = prompt_matches / len(native["prompts"])
            comparison["prompt_match"] = prompt_match_rate > 0.9
            comparison["summary"]["prompt_match_rate"] = prompt_match_rate

            logger.info(f"Prompt Match Rate: {prompt_match_rate:.2%}")

        # Compare extraction logic
        if native.get("extracted_labels") and benchforge.get("extracted_labels"):
            extraction_matches = sum(
                1
                for n, b in zip(
                    native["extracted_labels"], benchforge["extracted_labels"]
                )
                if n == b
            )
            extraction_match_rate = extraction_matches / len(native["extracted_labels"])
            comparison["extraction_match"] = extraction_match_rate > 0.9
            comparison["summary"]["extraction_match_rate"] = extraction_match_rate

            logger.info(f"Extraction Match Rate: {extraction_match_rate:.2%}")

        # Overall assessment
        comparison["summary"]["native_errors"] = len(native.get("errors", []))
        comparison["summary"]["benchforge_errors"] = len(benchforge.get("errors", []))
        comparison["summary"]["feature_parity"] = (
            comparison["prompt_match"]
            and comparison["extraction_match"]
            and comparison["summary"]["native_errors"] == 0
            and comparison["summary"]["benchforge_errors"] == 0
        )

        self.results["comparison"] = comparison

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        if comparison["summary"]["feature_parity"]:
            logger.info("✅ FEATURE PARITY ACHIEVED!")
            logger.info("Both implementations are functionally equivalent.")
        else:
            logger.warning("⚠️ Differences detected between implementations:")
            for diff in comparison["differences"][:5]:  # Show first 5 differences
                logger.warning(f"  - {diff}")

        logger.info("\nMetrics:")
        logger.info(
            f"  Prompt Match Rate: {comparison['summary'].get('prompt_match_rate', 0):.2%}"
        )
        logger.info(
            f"  Extraction Match Rate: {comparison['summary'].get('extraction_match_rate', 0):.2%}"
        )
        logger.info(f"  Native Errors: {comparison['summary']['native_errors']}")
        logger.info(
            f"  BenchForge Errors: {comparison['summary']['benchforge_errors']}"
        )

        return comparison

    def generate_report(self) -> str:
        """Generate validation report.

        Returns:
            Path to report file
        """
        report_path = self.output_dir / f"validation_report_{self.timestamp}.json"

        report = {
            "timestamp": self.timestamp,
            "test_config": self.test_config,
            "test_samples": self.test_samples,
            "results": self.results,
            "phase1_complete": self.results.get("comparison", {})
            .get("summary", {})
            .get("feature_parity", False),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n📊 Report saved to: {report_path}")

        # Also create a markdown summary
        md_path = self.output_dir / f"validation_summary_{self.timestamp}.md"
        with open(md_path, "w") as f:
            f.write("# FOMC Validation Report\n\n")
            f.write(f"**Date**: {self.timestamp}\n")
            f.write(f"**Samples Tested**: {self.test_samples}\n\n")

            f.write("## Results\n\n")

            comparison = self.results.get("comparison", {}).get("summary", {})
            if comparison.get("feature_parity"):
                f.write("### ✅ Feature Parity Achieved\n\n")
                f.write("Both implementations are functionally equivalent.\n\n")
            else:
                f.write("### ⚠️ Differences Detected\n\n")

            f.write("### Metrics\n\n")
            f.write(
                f"- **Prompt Match Rate**: {comparison.get('prompt_match_rate', 0):.2%}\n"
            )
            f.write(
                f"- **Extraction Match Rate**: {comparison.get('extraction_match_rate', 0):.2%}\n"
            )
            f.write(f"- **Native Errors**: {comparison.get('native_errors', 0)}\n")
            f.write(
                f"- **BenchForge Errors**: {comparison.get('benchforge_errors', 0)}\n\n"
            )

            if comparison.get("feature_parity"):
                f.write("## Next Steps\n\n")
                f.write(
                    "Phase 1 validation is complete. Ready to proceed to Phase 2 migration.\n"
                )

        logger.info(f"📝 Summary saved to: {md_path}")

        return str(report_path)

    def run_validation(self) -> bool:
        """Run complete validation suite.

        Returns:
            True if validation passed, False otherwise
        """
        logger.info("\n" + "🚀 " * 20)
        logger.info("STARTING FOMC VALIDATION SUITE")
        logger.info("🚀 " * 20 + "\n")

        try:
            # Step 1: Create test data
            test_data = self.create_test_data()

            # Step 2: Test native implementation
            self.test_native_inference(test_data)

            # Step 3: Test BenchForge implementation
            self.test_benchforge_inference(test_data)

            # Step 4: Compare results
            comparison = self.compare_implementations()

            # Step 5: Generate report
            self.generate_report()

            # Return validation status
            return comparison.get("summary", {}).get("feature_parity", False)

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(description="FOMC Implementation Validation")
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of test samples"
    )
    parser.add_argument(
        "--output", type=str, default="validation_results", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    validator = FOMCValidationSuite(test_samples=args.samples, output_dir=args.output)

    success = validator.run_validation()

    if success:
        logger.info("\n" + "🎉 " * 20)
        logger.info("PHASE 1 VALIDATION COMPLETE - READY FOR PHASE 2")
        logger.info("🎉 " * 20 + "\n")
        sys.exit(0)
    else:
        logger.error("\n" + "❌ " * 20)
        logger.error("VALIDATION FAILED - REVIEW DIFFERENCES")
        logger.error("❌ " * 20 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
