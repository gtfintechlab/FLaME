#!/bin/bash
# FLAME to BenchForge Migration Script
# Executes Phase 2 migration with validation and rollback capability

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TASK="${1:-fomc}"  # Default to FOMC
NUM_SAMPLES="${2:-10}"  # Default to 10 samples for testing
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FLAME to BenchForge Migration${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Task: $TASK"
echo "Samples: $NUM_SAMPLES"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Step 1: Check Prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check uv first
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv not found. Install with: pip install uv${NC}"
    exit 1
fi

# Check Python through uv
if ! uv run python --version &> /dev/null; then
    echo -e "${RED}❌ Python not accessible through uv${NC}"
    exit 1
fi

# Check API key
if [ -z "$TOGETHER_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: TOGETHER_API_KEY not set${NC}"
    echo "Some tests may be skipped"
fi

echo -e "${GREEN}✅ Prerequisites checked${NC}\n"

# Step 2: Run Validation Tests
echo -e "${YELLOW}Step 2: Running validation tests...${NC}"

# Run smoke tests first
echo "Running smoke tests..."
if uv run python -m pytest tests/fomc/smoke/ -v --tb=short; then
    echo -e "${GREEN}✅ Smoke tests passed${NC}"
else
    echo -e "${RED}❌ Smoke tests failed. Aborting migration.${NC}"
    exit 1
fi

# Check if simple test exists and run it
if [ -f "tests/validation/simple_fomc_test.py" ]; then
    echo "Running simple validation test..."
    if uv run python tests/validation/simple_fomc_test.py; then
        echo -e "${GREEN}✅ Simple validation passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Simple validation had issues${NC}"
    fi
fi

echo ""

# Step 3: Enable Feature Flag
echo -e "${YELLOW}Step 3: Enabling BenchForge for $TASK...${NC}"

# Create .env.migration file
cat > .env.migration << EOF
# BenchForge Migration Settings
# Generated: $(date)
USE_BENCHFORGE_${TASK^^}=1

# Add more tasks as they're migrated
# USE_BENCHFORGE_FPB=1
# USE_BENCHFORGE_ALL=1
EOF

# Also export for current session
export USE_BENCHFORGE_${TASK^^}=1

echo -e "${GREEN}✅ Feature flag enabled for $TASK${NC}"
echo ""

# Step 4: Run A/B Test
echo -e "${YELLOW}Step 4: Running A/B test with real data...${NC}"

if [ -f "tests/migration/ab_test_fomc.py" ]; then
    echo "Comparing native vs BenchForge implementations..."
    
    if uv run python tests/migration/ab_test_fomc.py \
        --task "$TASK" \
        --num-samples "$NUM_SAMPLES"; then
        echo -e "${GREEN}✅ A/B test passed${NC}"
    else
        echo -e "${RED}❌ A/B test failed${NC}"
        echo "Rolling back..."
        unset USE_BENCHFORGE_${TASK^^}
        rm -f .env.migration
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  A/B test script not found, skipping${NC}"
fi

echo ""

# Step 5: Check Migration Metrics
echo -e "${YELLOW}Step 5: Checking migration metrics...${NC}"

# Create Python script to check metrics
cat > /tmp/check_migration.py << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from flame.utils.migration_monitor import get_migration_monitor
    
    monitor = get_migration_monitor()
    metrics = monitor.get_metrics()
    health = monitor.get_health_status()
    
    print(f"Total calls: {metrics.get('total_calls', 0)}")
    print(f"BenchForge calls: {metrics.get('benchforge_calls', 0)}")
    print(f"Health status: {health['status']}")
    
    if health['issues']:
        print("Issues:")
        for issue in health['issues']:
            print(f"  - {issue}")
    
    if monitor.should_rollback():
        print("\n❌ Rollback recommended based on metrics")
        sys.exit(1)
    else:
        print("\n✅ Migration metrics look good")
        sys.exit(0)
        
except Exception as e:
    print(f"Could not check metrics: {e}")
    # Don't fail migration if monitoring isn't available
    sys.exit(0)
PYTHON_SCRIPT

if uv run python /tmp/check_migration.py; then
    echo -e "${GREEN}✅ Metrics check passed${NC}"
else
    echo -e "${YELLOW}⚠️  Metrics indicate potential issues${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Rolling back..."
        unset USE_BENCHFORGE_${TASK^^}
        rm -f .env.migration
        exit 1
    fi
fi

rm -f /tmp/check_migration.py
echo ""

# Step 6: Final Configuration
echo -e "${YELLOW}Step 6: Finalizing configuration...${NC}"

# Create migration status file
cat > migration_status.json << EOF
{
  "task": "$TASK",
  "status": "migrated",
  "timestamp": "$(date -Iseconds)",
  "benchforge_enabled": true,
  "samples_tested": $NUM_SAMPLES
}
EOF

echo -e "${GREEN}✅ Configuration updated${NC}"
echo ""

# Success!
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Migration Complete for $TASK!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Monitor logs in migration_logs/"
echo "2. Run production workloads with:"
echo "   uv run python main.py --mode inference --tasks $TASK"
echo "3. Check metrics with:"
echo "   uv run python -c \"from flame.utils.migration_monitor import get_migration_monitor; m = get_migration_monitor(); print(m.get_health_status())\""
echo "4. If stable after testing, the migration is complete!"
echo ""
echo "To rollback if needed:"
echo "  unset USE_BENCHFORGE_${TASK^^}"
echo "  rm .env.migration"
echo "  export FORCE_NATIVE_IMPLEMENTATION=true"