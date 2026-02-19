#!/bin/bash
# Run all tests in src/tests/ using the oasis conda environment
#
# Usage:
#   ./src/tests/run_tests.sh              # Run all tests (excludes outdated/)
#   ./src/tests/run_tests.sh -v           # Verbose output (default)
#   ./src/tests/run_tests.sh -k "kv"      # Run only tests matching "kv"
#   ./src/tests/run_tests.sh --no-tpu     # Skip TPU tests explicitly
#   ./src/tests/run_tests.sh --include-outdated  # Include outdated tests (may fail)
#
# Test files:
#   - test_attention_mask_implementations.py  # Mask correctness with get_token_info
#   - test_kv_cache_analysis.py               # KV cache + training/inference consistency
#   - test_splash_attention_outputs.py        # TPU kernel output accuracy
#   - test_mp_teacher_forcing_mask_bug.py     # MP teacher forcing bug detection
#
# Environment:
#   Uses conda environment "oasis" which has JAX and other dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# Default arguments
PYTEST_ARGS=("-v")
SKIP_TPU=false
INCLUDE_OUTDATED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-tpu)
            SKIP_TPU=true
            shift
            ;;
        --include-outdated)
            INCLUDE_OUTDATED=true
            shift
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Add TPU skip marker if requested
if [ "$SKIP_TPU" = true ]; then
    PYTEST_ARGS+=("-m" "not requires_tpu")
fi

# Exclude outdated tests by default (they have broken imports)
if [ "$INCLUDE_OUTDATED" = false ]; then
    PYTEST_ARGS+=("--ignore=src/tests/outdated")
fi

echo "========================================"
echo "Running solaris tests"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Conda env: oasis"
echo "Include outdated: $INCLUDE_OUTDATED"
echo "pytest args: ${PYTEST_ARGS[*]}"
echo "========================================"
echo

# Run pytest with the oasis conda environment
python -m pytest src/tests/ "${PYTEST_ARGS[@]}"

echo
echo "========================================"
echo "All tests completed!"
echo "========================================"
