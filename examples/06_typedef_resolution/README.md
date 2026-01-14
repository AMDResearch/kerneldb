# Example 06: Typedef Resolution

This example demonstrates how to resolve type aliases (typedef/using) including nested typedef chains.

```bash
python3 example.py
```

## Output

```terminal
Typedef Resolution Example
================================================================================

[1/3] Compiling HIP code with debug symbols...
✓ Compilation successful

[2/3] Analyzing with resolve_typedefs=False (default)...
     (Shows typedef/using names as written in source code)
Adding /tmp/tmps8cjvbp_
Found 1 kernels
Marker Count: 1 Kernel Count: 1

Kernel: typedef_kernel(int, float, int, int*, double, __hip_bfloat16, unsigned int)

  Arguments (7):
    a: MyInt (size=4, align=4)
    b: MyFloat (size=4, align=4)
    c: Level4 (size=4, align=4)
    d: IntPtr (size=8, align=8)
    e: MyTypedefDouble (size=8, align=8)
    g: bfloat16 (size=2, align=2)
    f: unsigned int (size=4, align=4)

[3/3] Analyzing with resolve_typedefs=True...

  Arguments (7):
    a: int (size=4, align=4)
    b: float (size=4, align=4)
    c: int (size=4, align=4)
    d: int* (size=8, align=8)
    e: double (size=8, align=8)
    g: struct __hip_bfloat16 (size=2, align=2)
    f: unsigned int (size=4, align=4)
Ending kernelDB
Found 1 kernels.
```
