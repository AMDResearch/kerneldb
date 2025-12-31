# Example 03: Kernel Arguments Analysis

This example demonstrates how to extract and display kernel argument (parameter) information from HIP kernels using kernelDB.

```bash
python3 example.py
```

## Output

```terminal
Kernel Arguments Example
================================================================================

[1/2] Compiling HIP code with debug symbols...
Compilation successful

[2/2] Analyzing kernel arguments...
Adding /tmp/tmpcaz6qmo0
Found 1 kernels
Marker Count: 1 Kernel Count: 1

Kernel: vector_add(double*, double*, double*, int)
  Arguments (4):
    a: double* (size=8, align=0)
    b: double* (size=8, align=0)
    c: double* (size=8, align=0)
    n: int (size=4, align=0)
Ending kernelDB
Found 1 kernels.
```