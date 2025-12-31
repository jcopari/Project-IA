#!/bin/bash
# Assembly Analysis Script for Qorus-IA v2.0
# Analyzes generated assembly code to verify optimizations

set -e

BUILD_DIR="build"
OUTPUT_DIR="assembly_analysis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Qorus-IA v2.0 Assembly Analysis Tool"
echo "======================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to analyze assembly file
analyze_assembly() {
    local source_file=$1
    local obj_file=$2
    local asm_file=$3
    
    echo "Analyzing: $source_file"
    
    # Generate assembly with Intel syntax and source annotations
    gcc -S -masm=intel -fverbose-asm -O3 -mavx2 -mfma \
        -std=c11 -I./include -D_GNU_SOURCE \
        "$source_file" -o "$asm_file" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not generate assembly for $source_file${NC}"
        return 1
    }
    
    # Check for AVX2 instructions
    if grep -q "vfmadd\|vmulps\|vaddps\|vsubps\|vmaxps\|vminps\|vrsqrtps" "$asm_file"; then
        echo -e "  ${GREEN}✓ AVX2 instructions detected${NC}"
    else
        echo -e "  ${RED}✗ No AVX2 instructions found${NC}"
    fi
    
    # Check for FMA instructions
    if grep -q "vfmadd" "$asm_file"; then
        echo -e "  ${GREEN}✓ FMA instructions detected${NC}"
    else
        echo -e "  ${YELLOW}⚠ No FMA instructions found${NC}"
    fi
    
    # Check for register spilling (indicates register pressure)
    if grep -q "rsp\|rbp" "$asm_file" | grep -q "\[rsp\|\[rbp"; then
        local spill_count=$(grep -c "\[rsp\|\[rbp" "$asm_file" || true)
        if [ "$spill_count" -gt 10 ]; then
            echo -e "  ${YELLOW}⚠ High register spilling detected ($spill_count instances)${NC}"
        else
            echo -e "  ${GREEN}✓ Low register spilling ($spill_count instances)${NC}"
        fi
    else
        echo -e "  ${GREEN}✓ No register spilling detected${NC}"
    fi
    
    # Check for loop unrolling
    if grep -q "\.L[0-9]*:" "$asm_file" | head -1; then
        local loop_count=$(grep -c "\.L[0-9]*:" "$asm_file" || true)
        if [ "$loop_count" -gt 5 ]; then
            echo -e "  ${GREEN}✓ Multiple loops detected (possible unrolling)${NC}"
        fi
    fi
    
    # Check for inlining (no function prologue/epilogue)
    if ! grep -q "push.*rbp\|mov.*rbp.*rsp" "$asm_file"; then
        echo -e "  ${GREEN}✓ Function appears to be inlined${NC}"
    fi
    
    echo ""
}

# Analyze critical kernels
echo "Analyzing Critical Kernels:"
echo "---------------------------"

# MatMul
if [ -f "src/ops/avx2/matmul.c" ]; then
    analyze_assembly "src/ops/avx2/matmul.c" \
                    "$BUILD_DIR/ops/avx2/matmul.o" \
                    "$OUTPUT_DIR/matmul.s"
fi

# RMSNorm
if [ -f "src/ops/avx2/rmsnorm.c" ]; then
    analyze_assembly "src/ops/avx2/rmsnorm.c" \
                    "$BUILD_DIR/ops/avx2/rmsnorm.o" \
                    "$OUTPUT_DIR/rmsnorm.s"
fi

# RoPE
if [ -f "src/ops/avx2/rope.c" ]; then
    analyze_assembly "src/ops/avx2/rope.c" \
                    "$BUILD_DIR/ops/avx2/rope.o" \
                    "$OUTPUT_DIR/rope.s"
fi

# SiLU
if [ -f "src/ops/avx2/silu.c" ]; then
    analyze_assembly "src/ops/avx2/silu.c" \
                    "$BUILD_DIR/ops/avx2/silu.o" \
                    "$OUTPUT_DIR/silu.s"
fi

# Softmax
if [ -f "src/ops/avx2/softmax.c" ]; then
    analyze_assembly "src/ops/avx2/softmax.c" \
                    "$BUILD_DIR/ops/avx2/softmax.o" \
                    "$OUTPUT_DIR/softmax.s"
fi

echo "Assembly files saved to: $OUTPUT_DIR/"
echo ""
echo "To view assembly with objdump:"
echo "  objdump -d -M intel $BUILD_DIR/ops/avx2/matmul.o | less"
echo ""
echo "To check for specific instructions:"
echo "  objdump -d $BUILD_DIR/ops/avx2/matmul.o | grep vfmadd"

