# Triton Kernel Analysis Example

Analyzes Triton-compiled GPU kernels - runs Triton code, locates compiled `.hsaco` files in cache, and maps assembly back to source.

```bash
module load pytorch  # Required for Triton
python3 example.py
```

## Output (AMD MI300X)

```
Triton Kernel Analysis Example
================================================================================

[1/3] Running Triton script to compile kernels...
Triton kernels executed

[2/3] Searching Triton cache for compiled kernels...
Found 47 objects, analyzing 3 most recent

[3/3] Analyzing kernels with kernelDB...
Found 2 kernel(s)

================================================================================
Kernel #1: add_kernel_0d1d2d3d4d5c6d7
Source: add_kernel_0d1d2d3d4d5c6d7.hsaco
================================================================================

Source lines: 9 (range: 15-38)
Basic blocks: 4

Instructions by source line:

  Line 15: 3 instruction(s)
    [15:5] 	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 00000000: C00A0000 00000000
    [15:5] 	s_load_dword s2, s[4:5], 0x8                               // 00000008: C0020080 00000008
    [15:5] 	s_waitcnt lgkmcnt(0)                                       // 00000010: BF8CC07F

  Line 17: 2 instruction(s)
    [17:18] 	v_mov_b32_e32 v1, s2                                       // 00000014: 7E020202
    [17:18] 	v_mul_lo_u32 v0, v0, v1                                    // 00000018: D2850000 00000300

  Line 18: 4 instruction(s)
    [18:20] 	v_add_u32_e32 v2, v0, v3                                   // 0000001C: 68040700
    [18:20] 	v_cmp_lt_u32_e32 vcc, v2, v4                               // 00000020: 7D880902
    [18:20] 	s_and_saveexec_b64 s[0:1], vcc                             // 00000024: BE80206A
    [18:20] 	s_cbranch_execz .L_skip                                    // 00000028: BF880008

  Line 21: 2 instruction(s)
    [21:9] 	global_load_dword v2, v[4:5], off                          // 0000002C: DC500000 027F0200
    [21:9] 	s_waitcnt vmcnt(0)                                         // 00000034: BF8C0F70

  Line 22: 2 instruction(s)
    [22:9] 	global_load_dword v3, v[6:7], off                          // 00000038: DC500000 037F0400
    [22:9] 	s_waitcnt vmcnt(0)                                         // 00000040: BF8C0F70

  Line 23: 1 instruction(s)
    [23:14] 	v_add_f32_e32 v2, v2, v3                                   // 00000044: 02040702

  Line 24: 1 instruction(s)
    [24:4] 	global_store_dword v[8:9], v2, off                         // 00000048: DC700000 027F0600

  Line 38: 1 instruction(s)
    [38:0] 	s_endpgm                                                   // 00000050: BF810000

Memory operations (load/store):
  	global_load_dword v2, v[4:5], off                          // 0000002C: DC500000 027F0200
  	global_load_dword v3, v[6:7], off                          // 00000038: DC500000 037F0400
  	global_store_dword v[8:9], v2, off                         // 00000048: DC700000 027F0600
  	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 00000000: C00A0000 00000000
Total: 5 memory operations

Source files:
  - /tmp/triton_add_kernel.py

================================================================================
Kernel #2: mul_kernel_0d1d2d3d4d5c6d7
Source: mul_kernel_0d1d2d3d4d5c6d7.hsaco
================================================================================

Source lines: 9 (range: 27-50)
Basic blocks: 4

Instructions by source line:

  Line 27: 3 instruction(s)
    [27:5] 	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 00000000: C00A0000 00000000
    [27:5] 	s_load_dword s2, s[4:5], 0x8                               // 00000008: C0020080 00000008
    [27:5] 	s_waitcnt lgkmcnt(0)                                       // 00000010: BF8CC07F

  Line 29: 2 instruction(s)
    [29:18] 	v_mov_b32_e32 v1, s2                                       // 00000014: 7E020202
    [29:18] 	v_mul_lo_u32 v0, v0, v1                                    // 00000018: D2850000 00000300

  Line 30: 4 instruction(s)
    [30:20] 	v_add_u32_e32 v2, v0, v3                                   // 0000001C: 68040700
    [30:20] 	v_cmp_lt_u32_e32 vcc, v2, v4                               // 00000020: 7D880902
    [30:20] 	s_and_saveexec_b64 s[0:1], vcc                             // 00000024: BE80206A
    [30:20] 	s_cbranch_execz .L_skip                                    // 00000028: BF880008

  Line 33: 2 instruction(s)
    [33:9] 	global_load_dword v2, v[4:5], off                          // 0000002C: DC500000 027F0200
    [33:9] 	s_waitcnt vmcnt(0)                                         // 00000034: BF8C0F70

  Line 34: 2 instruction(s)
    [34:9] 	global_load_dword v3, v[6:7], off                          // 00000038: DC500000 037F0400
    [34:9] 	s_waitcnt vmcnt(0)                                         // 00000040: BF8C0F70

  Line 35: 1 instruction(s)
    [35:14] 	v_mul_f32_e32 v2, v2, v3                                   // 00000044: 0A040702

  Line 36: 1 instruction(s)
    [36:4] 	global_store_dword v[8:9], v2, off                         // 00000048: DC700000 027F0600

  Line 50: 1 instruction(s)
    [50:0] 	s_endpgm                                                   // 00000050: BF810000

Memory operations (load/store):
  	global_load_dword v2, v[4:5], off                          // 0000002C: DC500000 027F0200
  	global_load_dword v3, v[6:7], off                          // 00000038: DC500000 037F0400
  	global_store_dword v[8:9], v2, off                         // 00000048: DC700000 027F0600
  	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 00000000: C00A0000 00000000
Total: 5 memory operations

Source files:
  - /tmp/triton_mul_kernel.py

================================================================================
Analysis complete!
```
