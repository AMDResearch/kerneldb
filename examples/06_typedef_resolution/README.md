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

Kernel: typedef_kernel(int, float, int, int*, double, __hip_bfloat16, unsigned int)

  Arguments (7):
    a: MyInt (size=0, align=0)
    b: MyFloat (size=0, align=0)
    c: Level4 (size=0, align=0)
    d: IntPtr (size=0, align=0)
    e: MyTypedefDouble (size=0, align=0)
    g: bfloat16 (size=0, align=0)
    f: unsigned int (size=4, align=4)

[3/3] Analyzing with resolve_typedefs=True...

  Arguments (7):
    a: int (size=0, align=0)
    b: float (size=0, align=0)
    c: int (size=0, align=0)
    d: int* (size=0, align=0)
    e: double (size=0, align=0)
    g: struct __hip_bfloat16 (size=0, align=0)
    f: unsigned int (size=4, align=4)
```
