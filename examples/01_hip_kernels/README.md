# HIP Kernel Analysis Example

Analyzes vector add and multiply HIP kernels - compiles code with debug symbols, extracts assembly instructions, and maps them back to source lines.

```bash
python3 example.py
```

## Output (AMD MI300X)

```
HIP Kernel Analysis Example
================================================================================

[1/2] Compiling HIP code with debug symbols...
Compilation successful

[2/2] Analyzing binary with kernelDB...
Adding /tmp/tmpy4sqcnqc
Found 2 kernels
Marker Count: 2 Kernel Count: 2
Found 2 kernel(s)

================================================================================
Kernel: vector_add(float*, float*, float*, int)
================================================================================

Source lines: 6 (range: 5-275)
Basic blocks: 3

Instructions by source line:

  Line 5: 2 instruction(s)
    [5:26]      s_mul_i32 s2, s2, s3                                       // 000000001F1C: 92020302
    [5:39]      v_add_u32_e32 v0, s2, v0                                   // 000000001F20: 68000002

  Line 6: 2 instruction(s)
    [6:13]      v_cmp_gt_i32_e32 vcc, s4, v0                               // 000000001F24: 7D880004
    [6:9]       s_and_saveexec_b64 s[2:3], vcc                             // 000000001F28: BE82206A

  Line 7: 6 instruction(s)
    [7:18]      global_load_dword v6, v[4:5], off                          // 000000001F60: DC508000 067F0004
    [7:27]      global_load_dword v7, v[2:3], off                          // 000000001F68: DC508000 077F0002
    [7:27]      v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]                   // 000000001F70: D2080000 04010002
    [7:25]      s_waitcnt vmcnt(0)                                         // 000000001F78: BF8C0F70
    [7:25]      v_add_f32_e32 v2, v6, v7                                   // 000000001F7C: 02040F06
    [7:16]      global_store_dword v[0:1], v2, off                         // 000000001F80: DC708000 007F0200

  Line 9: 1 instruction(s)
    [9:1]       s_endpgm                                                   // 000000001F88: BF810000

  Line 270: 8 instruction(s)
    [270:58]    s_load_dword s4, s[0:1], 0x18                              // 000000001F08: C0020100 00000018
    [270:58]    s_load_dwordx4 s[4:7], s[0:1], 0x0                         // 000000001F30: C00A0100 00000000
    [270:58]    s_load_dwordx2 s[2:3], s[0:1], 0x10                        // 000000001F38: C0060080 00000010
    [270:58]    v_ashrrev_i32_e32 v1, 31, v0                               // 000000001F40: 2202009F
    [270:58]    v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001F44: D28F0000 00020082
    [270:58]    s_waitcnt lgkmcnt(0)                                       // 000000001F4C: BF8CC07F
    [270:58]    v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]                   // 000000001F50: D2080004 04010004
    [270:58]    v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]                   // 000000001F58: D2080002 04010006

  Line 275: 3 instruction(s)
    [275:58]    s_load_dword s3, s[0:1], 0x2c                              // 000000001F00: C00200C0 0000002C
    [275:58]    s_waitcnt lgkmcnt(0)                                       // 000000001F10: BF8CC07F
    [275:58]    s_and_b32 s3, s3, 0xffff                                   // 000000001F14: 8603FF03 0000FFFF

Memory operations (load/store):
        global_load_dword v6, v[4:5], off                          // 000000001F60: DC508000 067F0004
        global_load_dword v7, v[2:3], off                          // 000000001F68: DC508000 077F0002
        global_store_dword v[0:1], v2, off                         // 000000001F80: DC708000 007F0200
        s_load_dword s4, s[0:1], 0x18                              // 000000001F08: C0020100 00000018
        s_load_dwordx4 s[4:7], s[0:1], 0x0                         // 000000001F30: C00A0100 00000000
        s_load_dwordx2 s[2:3], s[0:1], 0x10                        // 000000001F38: C0060080 00000010
        s_load_dword s3, s[0:1], 0x2c                              // 000000001F00: C00200C0 0000002C
Total: 7 memory operations

================================================================================
Kernel: vector_multiply(float*, float*, float*, int)
================================================================================

Source lines: 7 (range: 9-275)
Basic blocks: 3

Instructions by source line:

  Line 9: 29 instruction(s)
    [9:1]       s_nop 0                                                    // 000000001F8C: BF800000
    [9:1]       s_nop 0                                                    // 000000001F90: BF800000
    [9:1]       s_nop 0                                                    // 000000001F94: BF800000
    [9:1]       s_nop 0                                                    // 000000001F98: BF800000
    [9:1]       s_nop 0                                                    // 000000001F9C: BF800000
    [9:1]       s_nop 0                                                    // 000000001FA0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FA4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FA8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FAC: BF800000
    [9:1]       s_nop 0                                                    // 000000001FB0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FB4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FB8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FBC: BF800000
    [9:1]       s_nop 0                                                    // 000000001FC0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FC4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FC8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FCC: BF800000
    [9:1]       s_nop 0                                                    // 000000001FD0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FD4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FD8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FDC: BF800000
    [9:1]       s_nop 0                                                    // 000000001FE0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FE4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FE8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FEC: BF800000
    [9:1]       s_nop 0                                                    // 000000001FF0: BF800000
    [9:1]       s_nop 0                                                    // 000000001FF4: BF800000
    [9:1]       s_nop 0                                                    // 000000001FF8: BF800000
    [9:1]       s_nop 0                                                    // 000000001FFC: BF800000

  Line 12: 2 instruction(s)
    [12:26]     s_mul_i32 s2, s2, s3                                       // 00000000201C: 92020302
    [12:39]     v_add_u32_e32 v0, s2, v0                                   // 000000002020: 68000002

  Line 13: 2 instruction(s)
    [13:13]     v_cmp_gt_i32_e32 vcc, s4, v0                               // 000000002024: 7D880004
    [13:9]      s_and_saveexec_b64 s[2:3], vcc                             // 000000002028: BE82206A

  Line 14: 6 instruction(s)
    [14:18]     global_load_dword v6, v[4:5], off                          // 000000002060: DC508000 067F0004
    [14:27]     global_load_dword v7, v[2:3], off                          // 000000002068: DC508000 077F0002
    [14:27]     v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]                   // 000000002070: D2080000 04010002
    [14:25]     s_waitcnt vmcnt(0)                                         // 000000002078: BF8C0F70
    [14:25]     v_mul_f32_e32 v2, v6, v7                                   // 00000000207C: 0A040F06
    [14:16]     global_store_dword v[0:1], v2, off                         // 000000002080: DC708000 007F0200

  Line 16: 1 instruction(s)
    [16:1]      s_endpgm                                                   // 000000002088: BF810000

  Line 270: 8 instruction(s)
    [270:58]    s_load_dword s4, s[0:1], 0x18                              // 000000002008: C0020100 00000018
    [270:58]    s_load_dwordx4 s[4:7], s[0:1], 0x0                         // 000000002030: C00A0100 00000000
    [270:58]    s_load_dwordx2 s[2:3], s[0:1], 0x10                        // 000000002038: C0060080 00000010
    [270:58]    v_ashrrev_i32_e32 v1, 31, v0                               // 000000002040: 2202009F
    [270:58]    v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000002044: D28F0000 00020082
    [270:58]    s_waitcnt lgkmcnt(0)                                       // 00000000204C: BF8CC07F
    [270:58]    v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]                   // 000000002050: D2080004 04010004
    [270:58]    v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]                   // 000000002058: D2080002 04010006

  Line 275: 3 instruction(s)
    [275:58]    s_load_dword s3, s[0:1], 0x2c                              // 000000002000: C00200C0 0000002C
    [275:58]    s_waitcnt lgkmcnt(0)                                       // 000000002010: BF8CC07F
    [275:58]    s_and_b32 s3, s3, 0xffff                                   // 000000002014: 8603FF03 0000FFFF

Memory operations (load/store):
        global_load_dword v6, v[4:5], off                          // 000000002060: DC508000 067F0004
        global_load_dword v7, v[2:3], off                          // 000000002068: DC508000 077F0002
        global_store_dword v[0:1], v2, off                         // 000000002080: DC708000 007F0200
        s_load_dword s4, s[0:1], 0x18                              // 000000002008: C0020100 00000018
        s_load_dwordx4 s[4:7], s[0:1], 0x0                         // 000000002030: C00A0100 00000000
        s_load_dwordx2 s[2:3], s[0:1], 0x10                        // 000000002038: C0060080 00000010
        s_load_dword s3, s[0:1], 0x2c                              // 000000002000: C00200C0 0000002C
Total: 7 memory operations

================================================================================
Analysis complete!
Ending kernelDB
Found 2 kernels.
```
