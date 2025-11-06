# Triton Kernel Analysis Example

Analyzes Triton-compiled GPU kernels - runs Triton code, locates compiled `.hsaco` files in cache, and maps assembly back to source.

```bash
python3 example.py
```

## Output (AMD MI300X)

```
Triton Kernel Analysis Example
================================================================================

[1/3] Running Triton script to compile kernels...
Triton kernels executed

[2/3] Searching Triton cache for compiled kernels...
Found 43 objects, analyzing 3 most recent

[3/3] Analyzing kernels with kernelDB...
Adding /home1/muhaawad/.triton/cache/VEHFMMCU6N2LBQKTAEKAWK5SAY5KBMURLHDFWHCAD4YHKUE3BSMA/mul_kernel.hsaco
Found 1 kernels
Marker Count: 1 Kernel Count: 1
Adding /home1/muhaawad/.triton/cache/A3WQYMK4I2XDP2LIVP6XRUZ4FLJKLQ2P22A7YH6OV2MBT3SMQHYQ/add_kernel.hsaco
Found 1 kernels
Marker Count: 1 Kernel Count: 1
Adding /home1/muhaawad/.triton/cache/TWMT2S552FGTENXZDIMULA6TJBR757BDLU6DAUFRVNUT363GYWEQ/mul_kernel.hsaco
Found 1 kernels
Marker Count: 1 Kernel Count: 1
Ending kernelDB
Found 1 kernels.
Found 2 kernel(s)

================================================================================
Kernel #1: mul_kernel
Source: mul_kernel.hsaco
================================================================================

Source lines: 7 (range: 0-26)
Basic blocks: 10

Instructions by source line:

  Line 0: 12 instruction(s)
    [0:0]       s_lshl_b32 s0, s12, 10                                     // 000000001700: 8E008A0C
    [0:16]      v_lshl_add_u64 v[2:3], v[8:9], 2, s[2:3]                   // 000000001730: D2080002 00090508
    [0:16]      s_or_b64 exec, exec, s[0:1]                                // 000000001740: 87FE007E
    [0:16]      v_mov_b32_e32 v1, 0                                        // 000000001744: 7E020280
    [0:16]      v_mov_b32_e32 v2, 0                                        // 000000001748: 7E040280
    [0:16]      v_mov_b32_e32 v3, 0                                        // 00000000174C: 7E060280
    [0:16]      s_or_b64 exec, exec, s[0:1]                                // 000000001758: 87FE007E
    [0:4]       v_lshl_add_u64 v[0:1], v[8:9], 2, s[4:5]                   // 000000001768: D2080000 00110508
    [0:32]      v_lshl_add_u64 v[8:9], v[8:9], 2, s[6:7]                   // 000000001784: D2080008 00190508
    [0:32]      s_waitcnt vmcnt(0)                                         // 00000000178C: BF8C0F70
    [0:32]      v_pk_mul_f32 v[2:3], v[6:7], v[2:3]                        // 000000001790: D3B14002 18020506
    [0:32]      v_pk_mul_f32 v[0:1], v[4:5], v[0:1]                        // 000000001798: D3B14000 18020104

  Line 18: 61 instruction(s)
    [18:0]      s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001600: C0060080 00000000
    [18:0]      s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001608: C00E0100 00000008
    [18:0]      s_waitcnt lgkmcnt(0)                                       // 000000001610: BF8CC07F
    [18:0]      s_nop 0                                                    // 000000001618: BF800000
    [18:0]      s_nop 0                                                    // 00000000161C: BF800000
    [18:0]      s_nop 0                                                    // 000000001620: BF800000
    [18:0]      s_nop 0                                                    // 000000001624: BF800000
    [18:0]      s_nop 0                                                    // 000000001628: BF800000
    [18:0]      s_nop 0                                                    // 00000000162C: BF800000
    [18:0]      s_nop 0                                                    // 000000001630: BF800000
    [18:0]      s_nop 0                                                    // 000000001634: BF800000
    [18:0]      s_nop 0                                                    // 000000001638: BF800000
    [18:0]      s_nop 0                                                    // 00000000163C: BF800000
    [18:0]      s_nop 0                                                    // 000000001640: BF800000
    [18:0]      s_nop 0                                                    // 000000001644: BF800000
    [18:0]      s_nop 0                                                    // 000000001648: BF800000
    [18:0]      s_nop 0                                                    // 00000000164C: BF800000
    [18:0]      s_nop 0                                                    // 000000001650: BF800000
    [18:0]      s_nop 0                                                    // 000000001654: BF800000
    [18:0]      s_nop 0                                                    // 000000001658: BF800000
    [18:0]      s_nop 0                                                    // 00000000165C: BF800000
    [18:0]      s_nop 0                                                    // 000000001660: BF800000
    [18:0]      s_nop 0                                                    // 000000001664: BF800000
    [18:0]      s_nop 0                                                    // 000000001668: BF800000
    [18:0]      s_nop 0                                                    // 00000000166C: BF800000
    [18:0]      s_nop 0                                                    // 000000001670: BF800000
    [18:0]      s_nop 0                                                    // 000000001674: BF800000
    [18:0]      s_nop 0                                                    // 000000001678: BF800000
    [18:0]      s_nop 0                                                    // 00000000167C: BF800000
    [18:0]      s_nop 0                                                    // 000000001680: BF800000
    [18:0]      s_nop 0                                                    // 000000001684: BF800000
    [18:0]      s_nop 0                                                    // 000000001688: BF800000
    [18:0]      s_nop 0                                                    // 00000000168C: BF800000
    [18:0]      s_nop 0                                                    // 000000001690: BF800000
    [18:0]      s_nop 0                                                    // 000000001694: BF800000
    [18:0]      s_nop 0                                                    // 000000001698: BF800000
    [18:0]      s_nop 0                                                    // 00000000169C: BF800000
    [18:0]      s_nop 0                                                    // 0000000016A0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016A4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016A8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016AC: BF800000
    [18:0]      s_nop 0                                                    // 0000000016B0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016B4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016B8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016BC: BF800000
    [18:0]      s_nop 0                                                    // 0000000016C0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016C4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016C8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016CC: BF800000
    [18:0]      s_nop 0                                                    // 0000000016D0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016D4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016D8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016DC: BF800000
    [18:0]      s_nop 0                                                    // 0000000016E0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016E4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016E8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016EC: BF800000
    [18:0]      s_nop 0                                                    // 0000000016F0: BF800000
    [18:0]      s_nop 0                                                    // 0000000016F4: BF800000
    [18:0]      s_nop 0                                                    // 0000000016F8: BF800000
    [18:0]      s_nop 0                                                    // 0000000016FC: BF800000

  Line 21: 1 instruction(s)
    [21:28]     v_lshl_or_b32 v8, v0, 2, s0                                // 000000001704: D2000008 00010500

  Line 22: 1 instruction(s)
    [22:21]     v_cmp_gt_i32_e32 vcc, s8, v8                               // 00000000170C: 7D881008

  Line 23: 2 instruction(s)
    [23:16]     s_and_saveexec_b64 s[0:1], vcc                             // 000000001728: BE80206A
    [23:16]     global_load_dwordx4 v[4:7], v[2:3], off                    // 000000001738: DC5C8000 047F0002

  Line 24: 9 instruction(s)
    [24:16]     v_mov_b32_e32 v0, 0                                        // 000000001710: 7E000280
    [24:16]     v_ashrrev_i32_e32 v9, 31, v8                               // 000000001714: 2212109F
    [24:16]     v_mov_b32_e32 v4, 0                                        // 000000001718: 7E080280
    [24:16]     v_mov_b32_e32 v5, 0                                        // 00000000171C: 7E0A0280
    [24:16]     v_mov_b32_e32 v6, 0                                        // 000000001720: 7E0C0280
    [24:16]     v_mov_b32_e32 v7, 0                                        // 000000001724: 7E0E0280
    [24:16]     s_and_saveexec_b64 s[0:1], vcc                             // 000000001750: BE80206A
    [24:16]     global_load_dwordx4 v[0:3], v[0:1], off                    // 000000001770: DC5C8000 007F0000
    [24:16]     s_or_b64 exec, exec, s[0:1]                                // 000000001778: 87FE007E

  Line 26: 5 instruction(s)
    [26:32]     s_and_saveexec_b64 s[0:1], vcc                             // 00000000175C: BE80206A
    [26:4]      s_endpgm                                                   // 000000001764: BF810000
    [26:32]     s_and_saveexec_b64 s[0:1], vcc                             // 00000000177C: BE80206A
    [26:32]     global_store_dwordx4 v[8:9], v[0:3], off                   // 0000000017A0: DC7C8000 007F0008
    [26:4]      s_endpgm                                                   // 0000000017A8: BF810000

Memory operations (load/store):
        s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001600: C0060080 00000000
        s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001608: C00E0100 00000008
        global_load_dwordx4 v[4:7], v[2:3], off                    // 000000001738: DC5C8000 047F0002
        global_load_dwordx4 v[0:3], v[0:1], off                    // 000000001770: DC5C8000 007F0000
        global_store_dwordx4 v[8:9], v[0:3], off                   // 0000000017A0: DC7C8000 007F0008
Total: 5 memory operations
================================================================================
Kernel #2: add_kernel
Source: add_kernel.hsaco
================================================================================

Source lines: 7 (range: 0-14)
Basic blocks: 10

Instructions by source line:

  Line 0: 12 instruction(s)
    [0:0]       s_lshl_b32 s0, s12, 10                                     // 000000001700: 8E008A0C
    [0:16]      v_lshl_add_u64 v[2:3], v[8:9], 2, s[2:3]                   // 000000001730: D2080002 00090508
    [0:16]      s_or_b64 exec, exec, s[0:1]                                // 000000001740: 87FE007E
    [0:16]      v_mov_b32_e32 v1, 0                                        // 000000001744: 7E020280
    [0:16]      v_mov_b32_e32 v2, 0                                        // 000000001748: 7E040280
    [0:16]      v_mov_b32_e32 v3, 0                                        // 00000000174C: 7E060280
    [0:16]      s_or_b64 exec, exec, s[0:1]                                // 000000001758: 87FE007E
    [0:4]       v_lshl_add_u64 v[0:1], v[8:9], 2, s[4:5]                   // 000000001768: D2080000 00110508
    [0:32]      v_lshl_add_u64 v[8:9], v[8:9], 2, s[6:7]                   // 000000001784: D2080008 00190508
    [0:32]      s_waitcnt vmcnt(0)                                         // 00000000178C: BF8C0F70
    [0:32]      v_pk_add_f32 v[2:3], v[6:7], v[2:3]                        // 000000001790: D3B24002 18020506
    [0:32]      v_pk_add_f32 v[0:1], v[4:5], v[0:1]                        // 000000001798: D3B24000 18020104

  Line 6: 61 instruction(s)
    [6:0]       s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001600: C0060080 00000000
    [6:0]       s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001608: C00E0100 00000008
    [6:0]       s_waitcnt lgkmcnt(0)                                       // 000000001610: BF8CC07F
    [6:0]       s_nop 0                                                    // 000000001618: BF800000
    [6:0]       s_nop 0                                                    // 00000000161C: BF800000
    [6:0]       s_nop 0                                                    // 000000001620: BF800000
    [6:0]       s_nop 0                                                    // 000000001624: BF800000
    [6:0]       s_nop 0                                                    // 000000001628: BF800000
    [6:0]       s_nop 0                                                    // 00000000162C: BF800000
    [6:0]       s_nop 0                                                    // 000000001630: BF800000
    [6:0]       s_nop 0                                                    // 000000001634: BF800000
    [6:0]       s_nop 0                                                    // 000000001638: BF800000
    [6:0]       s_nop 0                                                    // 00000000163C: BF800000
    [6:0]       s_nop 0                                                    // 000000001640: BF800000
    [6:0]       s_nop 0                                                    // 000000001644: BF800000
    [6:0]       s_nop 0                                                    // 000000001648: BF800000
    [6:0]       s_nop 0                                                    // 00000000164C: BF800000
    [6:0]       s_nop 0                                                    // 000000001650: BF800000
    [6:0]       s_nop 0                                                    // 000000001654: BF800000
    [6:0]       s_nop 0                                                    // 000000001658: BF800000
    [6:0]       s_nop 0                                                    // 00000000165C: BF800000
    [6:0]       s_nop 0                                                    // 000000001660: BF800000
    [6:0]       s_nop 0                                                    // 000000001664: BF800000
    [6:0]       s_nop 0                                                    // 000000001668: BF800000
    [6:0]       s_nop 0                                                    // 00000000166C: BF800000
    [6:0]       s_nop 0                                                    // 000000001670: BF800000
    [6:0]       s_nop 0                                                    // 000000001674: BF800000
    [6:0]       s_nop 0                                                    // 000000001678: BF800000
    [6:0]       s_nop 0                                                    // 00000000167C: BF800000
    [6:0]       s_nop 0                                                    // 000000001680: BF800000
    [6:0]       s_nop 0                                                    // 000000001684: BF800000
    [6:0]       s_nop 0                                                    // 000000001688: BF800000
    [6:0]       s_nop 0                                                    // 00000000168C: BF800000
    [6:0]       s_nop 0                                                    // 000000001690: BF800000
    [6:0]       s_nop 0                                                    // 000000001694: BF800000
    [6:0]       s_nop 0                                                    // 000000001698: BF800000
    [6:0]       s_nop 0                                                    // 00000000169C: BF800000
    [6:0]       s_nop 0                                                    // 0000000016A0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016A4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016A8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016AC: BF800000
    [6:0]       s_nop 0                                                    // 0000000016B0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016B4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016B8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016BC: BF800000
    [6:0]       s_nop 0                                                    // 0000000016C0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016C4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016C8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016CC: BF800000
    [6:0]       s_nop 0                                                    // 0000000016D0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016D4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016D8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016DC: BF800000
    [6:0]       s_nop 0                                                    // 0000000016E0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016E4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016E8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016EC: BF800000
    [6:0]       s_nop 0                                                    // 0000000016F0: BF800000
    [6:0]       s_nop 0                                                    // 0000000016F4: BF800000
    [6:0]       s_nop 0                                                    // 0000000016F8: BF800000
    [6:0]       s_nop 0                                                    // 0000000016FC: BF800000

  Line 9: 1 instruction(s)
    [9:28]      v_lshl_or_b32 v8, v0, 2, s0                                // 000000001704: D2000008 00010500

  Line 10: 1 instruction(s)
    [10:21]     v_cmp_gt_i32_e32 vcc, s8, v8                               // 00000000170C: 7D881008

  Line 11: 2 instruction(s)
    [11:16]     s_and_saveexec_b64 s[0:1], vcc                             // 000000001728: BE80206A
    [11:16]     global_load_dwordx4 v[4:7], v[2:3], off                    // 000000001738: DC5C8000 047F0002

  Line 12: 9 instruction(s)
    [12:16]     v_mov_b32_e32 v0, 0                                        // 000000001710: 7E000280
    [12:16]     v_ashrrev_i32_e32 v9, 31, v8                               // 000000001714: 2212109F
    [12:16]     v_mov_b32_e32 v4, 0                                        // 000000001718: 7E080280
    [12:16]     v_mov_b32_e32 v5, 0                                        // 00000000171C: 7E0A0280
    [12:16]     v_mov_b32_e32 v6, 0                                        // 000000001720: 7E0C0280
    [12:16]     v_mov_b32_e32 v7, 0                                        // 000000001724: 7E0E0280
    [12:16]     s_and_saveexec_b64 s[0:1], vcc                             // 000000001750: BE80206A
    [12:16]     global_load_dwordx4 v[0:3], v[0:1], off                    // 000000001770: DC5C8000 007F0000
    [12:16]     s_or_b64 exec, exec, s[0:1]                                // 000000001778: 87FE007E

  Line 14: 5 instruction(s)
    [14:32]     s_and_saveexec_b64 s[0:1], vcc                             // 00000000175C: BE80206A
    [14:4]      s_endpgm                                                   // 000000001764: BF810000
    [14:32]     s_and_saveexec_b64 s[0:1], vcc                             // 00000000177C: BE80206A
    [14:32]     global_store_dwordx4 v[8:9], v[0:3], off                   // 0000000017A0: DC7C8000 007F0008
    [14:4]      s_endpgm                                                   // 0000000017A8: BF810000

Memory operations (load/store):
        s_load_dwordx2 s[2:3], s[0:1], 0x0                         // 000000001600: C0060080 00000000
        s_load_dwordx8 s[4:11], s[0:1], 0x8                        // 000000001608: C00E0100 00000008
        global_load_dwordx4 v[4:7], v[2:3], off                    // 000000001738: DC5C8000 047F0002
        global_load_dwordx4 v[0:3], v[0:1], off                    // 000000001770: DC5C8000 007F0000
        global_store_dwordx4 v[8:9], v[0:3], off                   // 0000000017A0: DC7C8000 007F0008
Total: 5 memory operations

================================================================================
Analysis complete!
Ending kernelDB
Found 1 kernels.
Ending kernelDB
Found 1 kernels.
```
