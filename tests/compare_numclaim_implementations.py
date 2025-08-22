#!/usr/bin/env python3
"""Compare NumClaim implementations: BenchForge vs Native FLAME.

This script:
1. Runs NumClaim with BenchForge implementation
2. Runs NumClaim with native FLAME implementation
3. Compares results and metrics
4. Tests with live API calls
"""

import os
import sys
import time
import json
import logging
import pandas as pd
from typing import Dict, List, Any
from litellm import completion

# Add paths for both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_benchforge_numclaim(samples: List[Dict], model: str = "gpt-3.5-turbo"):
    """Run NumClaim using BenchForge implementation."""
    print("\n" + "="*60)
    print("BENCHFORGE IMPLEMENTATION")
    print("="*60)
    
    from bench_forge.flame.tasks.numclaim import NumClaimTask, NumClaimConfig
    from bench_forge.tasks.config import PromptFormat
    
    # Configure task
    config = NumClaimConfig(
        name="numclaim",
        model=model,
        prompt_format=PromptFormat.FEW_SHOT
    )
    task = NumClaimTask(config)
    
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}/{len(samples)}")
        print(f"Text: {sample['context'][:80]}...")
        print(f"Expected: {sample.get('response', 'N/A')}")
        
        # Create prompt
        prompt = task.create_prompt(sample)
        
        try:
            # Make API call
            start_time = time.time()
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            api_time = time.time() - start_time
            
            # Extract response
            response_text = response.choices[0].message.content
            print(f"Raw response: {response_text[:100]}")
            
            # Extract label
            extracted = task.extract_label_from_response(response_text)
            print(f"Extracted: {extracted}")
            print(f"Time: {api_time:.2f}s")
            
            results.append({
                "implementation": "BenchForge",
                "sample_id": i,
                "context": sample['context'],
                "expected": sample.get('response'),
                "raw_response": response_text,
                "extracted": extracted,
                "api_time": api_time
            })
            
        except Exception as e:
            logger.error(f"BenchForge error on sample {i}: {e}")
            results.append({
                "implementation": "BenchForge",
                "sample_id": i,
                "context": sample['context'],
                "expected": sample.get('response'),
                "raw_response": str(e),
                "extracted": None,
                "api_time": 0
            })
    
    return results


def run_flame_numclaim(samples: List[Dict], model: str = "gpt-3.5-turbo"):
    """Run NumClaim using native FLAME implementation."""
    print("\n" + "="*60)
    print("NATIVE FLAME IMPLEMENTATION")
    print("="*60)
    
    try:
        # Import FLAME modules
        from src.flame.code.prompts import PromptFormat as FLAMEPromptFormat, get_prompt
        from litellm import completion
        
        # Get FLAME prompt
        numclaim_prompt = get_prompt("numclaim", FLAMEPromptFormat.ZERO_SHOT)
        if numclaim_prompt is None:
            # Try to import and register prompts
            import src.flame.code.prompts.zeroshot
            import src.flame.code.prompts.fewshot
            numclaim_prompt = get_prompt("numclaim", FLAMEPromptFormat.ZERO_SHOT)
        
        if numclaim_prompt is None:
            logger.error("Could not load FLAME NumClaim prompt")
            return []
        
        results = []
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}/{len(samples)}")
            print(f"Text: {sample['context'][:80]}...")
            print(f"Expected: {sample.get('response', 'N/A')}")
            
            # Create prompt using FLAME format
            prompt = numclaim_prompt(sample['context'])
            
            try:
                # Make API call
                start_time = time.time()
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=100
                )
                api_time = time.time() - start_time
                
                # Extract response
                response_text = response.choices[0].message.content
                print(f"Raw response: {response_text[:100]}")
                
                # Extract label using FLAME logic
                # FLAME expects the label in the first line
                lines = response_text.strip().split('\n')
                extracted = None
                if lines:
                    first_line = lines[0].strip().upper()
                    if "INCLAIM" in first_line:
                        extracted = "INCLAIM"
                    elif "OUTOFCLAIM" in first_line:
                        extracted = "OUTOFCLAIM"
                
                print(f"Extracted: {extracted}")
                print(f"Time: {api_time:.2f}s")
                
                results.append({
                    "implementation": "FLAME",
                    "sample_id": i,
                    "context": sample['context'],
                    "expected": sample.get('response'),
                    "raw_response": response_text,
                    "extracted": extracted,
                    "api_time": api_time
                })
                
            except Exception as e:
                logger.error(f"FLAME API error on sample {i}: {e}")
                results.append({
                    "implementation": "FLAME",
                    "sample_id": i,
                    "context": sample['context'],
                    "expected": sample.get('response'),
                    "raw_response": str(e),
                    "extracted": None,
                    "api_time": 0
                })
        
        return results
        
    except ImportError as e:
        logger.error(f"Could not import FLAME modules: {e}")
        return []


def compare_results(benchforge_results: List[Dict], flame_results: List[Dict]):
    """Compare results from both implementations."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if not benchforge_results or not flame_results:
        print("⚠️ Missing results from one or both implementations")
        return
    
    # Create DataFrames for analysis
    bf_df = pd.DataFrame(benchforge_results)
    fl_df = pd.DataFrame(flame_results)
    
    # Calculate metrics for BenchForge
    bf_correct = sum(1 for r in benchforge_results 
                     if r['extracted'] == r['expected'] and r['extracted'] is not None)
    bf_extracted = sum(1 for r in benchforge_results if r['extracted'] is not None)
    bf_total = len(benchforge_results)
    
    # Calculate metrics for FLAME
    fl_correct = sum(1 for r in flame_results 
                     if r['extracted'] == r['expected'] and r['extracted'] is not None)
    fl_extracted = sum(1 for r in flame_results if r['extracted'] is not None)
    fl_total = len(flame_results)
    
    print("\n📊 METRICS COMPARISON:")
    print("-" * 40)
    print(f"{'Metric':<25} {'BenchForge':<15} {'FLAME':<15}")
    print("-" * 40)
    print(f"{'Total Samples':<25} {bf_total:<15} {fl_total:<15}")
    print(f"{'Extraction Success':<25} {bf_extracted}/{bf_total} ({bf_extracted/bf_total*100:.1f}%){'':5} "
          f"{fl_extracted}/{fl_total} ({fl_extracted/fl_total*100:.1f}%)")
    print(f"{'Accuracy':<25} {bf_correct}/{bf_total} ({bf_correct/bf_total*100:.1f}%){'':5} "
          f"{fl_correct}/{fl_total} ({fl_correct/fl_total*100:.1f}%)")
    
    # Average API time
    bf_avg_time = bf_df['api_time'].mean()
    fl_avg_time = fl_df['api_time'].mean()
    print(f"{'Avg API Time (s)':<25} {bf_avg_time:.3f}{'':10} {fl_avg_time:.3f}")
    
    # Compare individual samples
    print("\n📝 SAMPLE-BY-SAMPLE COMPARISON:")
    print("-" * 60)
    
    for i in range(min(len(benchforge_results), len(flame_results))):
        bf_result = benchforge_results[i]
        fl_result = flame_results[i]
        
        print(f"\nSample {bf_result['sample_id']}:")
        print(f"  Context: {bf_result['context'][:60]}...")
        print(f"  Expected: {bf_result['expected']}")
        print(f"  BenchForge: {bf_result['extracted']} {'✅' if bf_result['extracted'] == bf_result['expected'] else '❌'}")
        print(f"  FLAME: {fl_result['extracted']} {'✅' if fl_result['extracted'] == fl_result['expected'] else '❌'}")
        
        if bf_result['extracted'] != fl_result['extracted']:
            print(f"  ⚠️ DISAGREEMENT! BenchForge: {bf_result['extracted']}, FLAME: {fl_result['extracted']}")
    
    # Check for parity
    disagreements = sum(1 for i in range(min(len(benchforge_results), len(flame_results)))
                       if benchforge_results[i]['extracted'] != flame_results[i]['extracted'])
    
    print("\n" + "="*60)
    print("PARITY CHECK")
    print("="*60)
    
    if disagreements == 0:
        print("✅ PERFECT PARITY: Both implementations produce identical results!")
    else:
        print(f"⚠️ FOUND {disagreements} DISAGREEMENTS between implementations")
    
    # Performance comparison
    if bf_extracted > 0 and fl_extracted > 0:
        if bf_extracted/bf_total >= fl_extracted/fl_total:
            print(f"✅ BenchForge extraction rate ({bf_extracted/bf_total*100:.1f}%) >= FLAME ({fl_extracted/fl_total*100:.1f}%)")
        else:
            print(f"⚠️ BenchForge extraction rate ({bf_extracted/bf_total*100:.1f}%) < FLAME ({fl_extracted/fl_total*100:.1f}%)")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print(" NUMCLAIM IMPLEMENTATION COMPARISON: BENCHFORGE vs FLAME")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        return 1
    
    # Test samples covering various cases
    test_samples = [
        {
            "context": "The company reported revenue of $3.2 billion, representing a 25% increase year-over-year.",
            "response": "INCLAIM",
            "description": "Clear numerical claim with revenue and percentage"
        },
        {
            "context": "Market conditions continue to improve across all segments.",
            "response": "OUTOFCLAIM",
            "description": "General statement without numbers"
        },
        {
            "context": "Our operating margin expanded to 23.5%, exceeding analyst expectations.",
            "response": "INCLAIM",
            "description": "Percentage claim"
        },
        {
            "context": "The board remains confident in the company's strategic direction.",
            "response": "OUTOFCLAIM",
            "description": "Qualitative statement"
        },
        {
            "context": "We added 1.2 million new subscribers this quarter, bringing total to 45 million.",
            "response": "INCLAIM",
            "description": "Multiple numerical claims"
        },
        {
            "context": "Innovation remains at the core of our business strategy.",
            "response": "OUTOFCLAIM",
            "description": "Strategic statement without numbers"
        },
        {
            "context": "Free cash flow reached $850 million, up from $620 million last year.",
            "response": "INCLAIM",
            "description": "Financial comparison with numbers"
        },
        {
            "context": "We expect continued momentum in the coming quarters.",
            "response": "OUTOFCLAIM",
            "description": "Forward-looking without specifics"
        }
    ]
    
    print(f"\n📊 Testing with {len(test_samples)} diverse samples...")
    
    # Run BenchForge implementation
    benchforge_results = run_benchforge_numclaim(test_samples)
    
    # Run FLAME implementation
    flame_results = run_flame_numclaim(test_samples)
    
    # Compare results
    compare_results(benchforge_results, flame_results)
    
    # Save results to file
    results_file = "numclaim_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "benchforge": benchforge_results,
            "flame": flame_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())