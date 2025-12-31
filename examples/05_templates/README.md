# Example 05: Template Kernels

This example demonstrates how kernelDB discovers C++ template kernel instantiations.

```bash
python example.py
```

## Output

```
Template Kernels Example
================================================================================

[1/2] Compiling HIP code...
Compilation successful

[2/2] Analyzing template instantiations...
Adding /tmp/tmpj_192gl9
Found 3 kernels
Marker Count: 3 Kernel Count: 3
Found 3 template instantiation(s)

Kernel: void scale_values<double>(double*, double*, double, int)
  Arguments (4):
    input: double* (size=8, align=8)
    output: double* (size=8, align=8)
    factor: double (size=8, align=8)
    n: int (size=4, align=4)

Kernel: void scale_values<float>(float*, float*, float, int)
  Arguments (4):
    input: float* (size=8, align=8)
    output: float* (size=8, align=8)
    factor: float (size=4, align=4)
    n: int (size=4, align=4)

Kernel: void scale_values<int>(int*, int*, int, int)
  Arguments (4):
    input: int* (size=8, align=8)
    output: int* (size=8, align=8)
    factor: int (size=4, align=4)
    n: int (size=4, align=4)

Ending kernelDB
Found 3 kernels.
```
