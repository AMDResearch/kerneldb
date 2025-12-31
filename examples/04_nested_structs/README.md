# Example 04: Nested Structs

This example demonstrates how kernelDB handles nested struct kernel arguments including:
- Nested struct types (structs containing other structs)
- Recursive member extraction showing the full hierarchy

```bash
python example.py
```

## Output

```
Nested Structs Example
================================================================================

[1/2] Compiling HIP code with debug symbols...
Compilation successful

[2/2] Analyzing kernel arguments...
Adding /tmp/tmpr1j6513k
Found 1 kernels
Marker Count: 1 Kernel Count: 1
Found 1 kernel(s)


================================================================================
Kernel: update_particles(Particle*, BoundingBox, int)
================================================================================

Arguments (3):

  [1] particles: struct Particle*
      Size: 8 bytes, Alignment: 8 bytes

  [2] bounds: struct BoundingBox
      Size: 24 bytes, Alignment: 4 bytes
      Struct members:
        min: struct Point3D (size=12, offset=0)
          Members:
            x: float (size=4, offset=0)
            y: float (size=4, offset=4)
            z: float (size=4, offset=8)
        max: struct Point3D (size=12, offset=12)
          Members:
            x: float (size=4, offset=0)
            y: float (size=4, offset=4)
            z: float (size=4, offset=8)

  [3] count: int
      Size: 4 bytes, Alignment: 4 bytes
Ending kernelDB
Found 1 kernels.
```
