#!/usr/bin/env python3
"""Apply performance and extraction fixes to BenchForge.

This script applies two critical fixes:
1. Performance: Use true parallel batch processing with litellm
2. Extraction: Improve label extraction logic for better success rate
"""

import shutil
from pathlib import Path
import sys


def apply_fixes():
    """Apply the performance and extraction fixes."""

    print("=" * 60)
    print("BenchForge Performance & Extraction Fix")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("benchforge").exists():
        print("ERROR: benchforge directory not found. Run this from FLAME root.")
        return False

    # Fix 1: Replace LLM client with parallel version
    print("\n1. Applying performance fix (parallel batch processing)...")

    client_original = Path("benchforge/bench_forge/llm/client.py")
    client_fixed = Path("benchforge/bench_forge/llm/client_fixed.py")
    client_backup = Path("benchforge/bench_forge/llm/client_original.py")

    if client_fixed.exists():
        # Backup original if not already done
        if not client_backup.exists():
            shutil.copy2(client_original, client_backup)
            print(f"   Backed up original to {client_backup}")

        # Apply fix
        shutil.copy2(client_fixed, client_original)
        print(f"   ✅ Applied parallel batch processing fix to {client_original}")
    else:
        print(f"   ❌ Fixed client not found at {client_fixed}")
        return False

    # Fix 2: Replace FOMC task with improved extraction
    print("\n2. Applying extraction fix (improved label extraction)...")

    fomc_original = Path("benchforge/bench_forge/flame/tasks/fomc.py")
    fomc_fixed = Path("benchforge/bench_forge/flame/tasks/fomc_fixed.py")
    fomc_backup = Path("benchforge/bench_forge/flame/tasks/fomc_original.py")

    if fomc_fixed.exists():
        # Backup original if not already done
        if not fomc_backup.exists():
            shutil.copy2(fomc_original, fomc_backup)
            print(f"   Backed up original to {fomc_backup}")

        # Apply fix
        shutil.copy2(fomc_fixed, fomc_original)
        print(f"   ✅ Applied improved extraction logic to {fomc_original}")
    else:
        print(f"   ❌ Fixed FOMC task not found at {fomc_fixed}")
        return False

    print("\n" + "=" * 60)
    print("✅ FIXES APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nExpected improvements:")
    print("- Performance: ~5x faster (from 430s to ~85s for 496 samples)")
    print("- Extraction: >95% success rate (from 20% to >95%)")
    print("\nTo test the fixes, run:")
    print("  python run_full_migration_test.py")
    print("\nTo revert changes, run:")
    print(
        "  cp benchforge/bench_forge/llm/client_original.py benchforge/bench_forge/llm/client.py"
    )
    print(
        "  cp benchforge/bench_forge/flame/tasks/fomc_original.py benchforge/bench_forge/flame/tasks/fomc.py"
    )
    print("=" * 60)

    return True


def main():
    """Main entry point."""
    success = apply_fixes()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
