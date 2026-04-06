#!/bin/bash
# Build the detour_nav Python extension module.
#
# Prerequisites:
#   pip install pybind11
#   cmake >= 3.15, a C++17 compiler
#
# Usage:
#   cd detour_nav && bash build.sh
#
# The built .so file is copied to the project root for easy importing.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"

make -j"$(nproc)"

# Copy the built module to the project root
cp detour_nav*.so "$PROJECT_ROOT/"
echo "Built detour_nav module -> $PROJECT_ROOT/detour_nav$(python3-config --extension-suffix 2>/dev/null || echo '.so')"
