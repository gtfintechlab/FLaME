#!/bin/bash
# Rollback script for BenchForge migration
# Immediately reverts to native implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TASK="${1:-all}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}BenchForge Migration Rollback${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Task: $TASK"
echo ""

# Step 1: Disable BenchForge
echo -e "${BLUE}Step 1: Disabling BenchForge...${NC}"

if [ "$TASK" = "all" ]; then
    # Unset all BenchForge flags
    unset USE_BENCHFORGE_ALL
    unset USE_BENCHFORGE_FOMC
    unset USE_BENCHFORGE_FPB
    unset USE_BENCHFORGE_HEADLINE
    unset USE_BENCHFORGE_NER
    echo "Disabled all BenchForge tasks"
else
    # Unset specific task
    TASK_UPPER=$(echo "$TASK" | tr '[:lower:]' '[:upper:]')
    unset USE_BENCHFORGE_${TASK_UPPER}
    echo "Disabled BenchForge for $TASK"
fi

# Step 2: Force native implementation
echo -e "${BLUE}Step 2: Forcing native implementation...${NC}"
export FORCE_NATIVE_IMPLEMENTATION=true
echo "FORCE_NATIVE_IMPLEMENTATION=true"

# Step 3: Remove migration configuration
echo -e "${BLUE}Step 3: Removing migration configuration...${NC}"

if [ -f ".env.migration" ]; then
    mv .env.migration .env.migration.rollback.$(date +%Y%m%d_%H%M%S)
    echo "Backed up .env.migration"
fi

if [ -f "migration_status.json" ]; then
    mv migration_status.json migration_status.rollback.$(date +%Y%m%d_%H%M%S).json
    echo "Backed up migration_status.json"
fi

# Step 4: Create rollback marker
echo -e "${BLUE}Step 4: Creating rollback marker...${NC}"

cat > rollback_status.json << EOF
{
  "task": "$TASK",
  "status": "rolled_back",
  "timestamp": "$(date -Iseconds)",
  "reason": "Manual rollback initiated",
  "native_forced": true
}
EOF

echo -e "${GREEN}✅ Rollback marker created${NC}"

# Success
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Rollback Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Native implementation is now active."
echo ""
echo "To re-enable BenchForge later:"
echo "  unset FORCE_NATIVE_IMPLEMENTATION"
echo "  export USE_BENCHFORGE_${TASK^^}=1"
echo ""
echo "To check current status:"
echo "  python -c \"from flame.migration_config import MIGRATION_CONFIG; print(MIGRATION_CONFIG.get_status())\""