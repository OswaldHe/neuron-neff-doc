# Worked example: every slot of HCCollapse `sg00`

Generated from `artifacts/corpus/HCCollapse.parquet`. *fields* lists the non-zero decoded fields (vendor names, `tools/isa.py`), with union aliases and reserved bytes left out. *line* is the NKI source line from the debug info; the source is `optimization-runs-HCCollapse/hc_collapse-full/source.py`.

## SP

| slot | opcode | BIR name | line | source | fields |
|---|---|---|---|---|---|
| 0 | `SET_ORDERING_MODE` | nan |  | `` | events_wait_mode=NONE; events_update_mode=NONE |
| 1 | `PSEUDO_DMATRIGGER` | I-6_inst__I-4-0-trig | 145 | `nisa.dma_copy(dst=xg, src=x3[:, 0:hc, f0:f0 + w], dge_mode=nisa.dge_mo` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0 |
| 2 | `PSEUDO_DMATRIGGER` | I-0_inst__I-4-0-trig | 138 | `nisa.dma_copy(dst=pm, src=pm2, dge_mode=nisa.dge_mode.none)` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0; count_union_block_id=1 |
| 3 | `PSEUDO_DMATRIGGER` | I-1_inst__I-4-0-trig | 145 | `nisa.dma_copy(dst=xg, src=x3[:, 0:hc, f0:f0 + w], dge_mode=nisa.dge_mo` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0; count_union_block_id=2 |
| 4 | `EVENT_SEMAPHORE` | I-11_inst__I-4-0-trig-wait-0 | 160 | `nisa.dma_copy(dst=cc_src, src=yg, dge_mode=nisa.dge_mode.none)` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=24; events_update_mode=NONE; events_semaphore_value=1; events_extended_wait_mode=NONE; events_extended_update_mode=NONE |
| 5 | `PSEUDO_DMATRIGGER` | I-11_inst__I-4-0-trig | 160 | `nisa.dma_copy(dst=cc_src, src=yg, dge_mode=nisa.dge_mode.none)` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=NONE; events_semaphore_value=7; dma_queue_name=qSPSpillReload0 |
| 6 | `PSEUDO_DMATRIGGER` | I-13_inst__I-4-0-trig | 172 | `nisa.dma_copy(dst=y2[:, r * shard:(r + 1) * shard],` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=8; events_update_mode=NONE; events_semaphore_value=1; dma_queue_name=qSPIO0; count_union_block_id=3 |
| 7 | `PSEUDO_DMATRIGGER` | I-14_inst__I-4-0-trig | 172 | `nisa.dma_copy(dst=y2[:, r * shard:(r + 1) * shard],` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0; count_union_block_id=4 |
| 8 | `PSEUDO_DMATRIGGER` | I-15_inst__I-4-0-trig | 172 | `nisa.dma_copy(dst=y2[:, r * shard:(r + 1) * shard],` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0; count_union_block_id=5 |
| 9 | `PSEUDO_DMATRIGGER` | I-16_inst__I-4-0-trig | 172 | `nisa.dma_copy(dst=y2[:, r * shard:(r + 1) * shard],` | events_wait_mode=NONE; events_update_mode=NONE; dma_queue_name=qSPIO0; count_union_block_id=6 |
| 10 | `NOP` | Block1-final-DMA-NOP | 172 | `nisa.dma_copy(dst=y2[:, r * shard:(r + 1) * shard],` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=32; events_update_mode=NONE; events_semaphore_value=16; notification_kind=NONE; hint_or_notific_inst_flush_flags_flush_mode=FULL; hint_or_notific_custom_notific_lo_src=REGISTER; hint_or_notific_custom_notific_hi_src=REGISTER |

## Pool

| slot | opcode | BIR name | line | source | fields |
|---|---|---|---|---|---|
| 0 | `SET_ORDERING_MODE` | nan |  | `` | events_wait_mode=NONE; events_update_mode=NONE |
| 1 | `PSEUDO_CORE_BARRIER` | cb_sg00_Block1_start |  | `` | events_wait_mode=NONE; events_update_mode=SEM_INC_COMPLETE; events_update_idx=24; semaphore=25 |
| 2 | `PSEUDO_CORE_BARRIER` | AG-0_inst__I-4-0_llc_cb_before_global_cc |  | `` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=42; events_update_mode=SEM_INC_COMPLETE; events_update_idx=24; events_semaphore_value=16; id=1; semaphore=25 |
| 3 | `PSEUDO_TRIGGER_COLLECTIVE2` | AG-0_inst__I-4-0 | 164 | `ncc.all_gather(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_gro` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=24; events_update_mode=SEM_INC_COMPLETE; events_update_idx=8; events_semaphore_value=2; op=BYPASS; dtype=BFLOAT16; ctype=ALL_GATHER; cc_dim=DIM_0; src0_start_addr_addr_immediate=1152921521786716160; src0_start_addr_addr_table_table_index=4; src0_start_addr_addr_reg_marker=16; src0_step_elem=[1, 1]; src0_num_elem=[163840, 1] |
| 4 | `PSEUDO_EXTENSION` | AG-0_inst__I-4-0 | 164 | `ncc.all_gather(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_gro` | opcode=PSEUDO_EXTENSION; instr_chaining=INSTR_CHAINING_NONE; dst_start_addr_addr_immediate=1152921526081683456; dst_start_addr_addr_table_table_index=5; dst_start_addr_addr_reg_marker=16; dst_step_elem=[1, 1]; dst_num_elem=[655360, 1]; unique_tensors=UNIQUE_TENSORS_UNKNOWN; permute_chain=PERMUTE_CHAIN_OFF; cross_sb_output_transfer=CROSS_SB_OUTPUT_TRANSFER_OFF |
| 5 | `PSEUDO_CORE_BARRIER` | cb_sg00_Block1_end |  | `` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=8; events_update_mode=SEM_INC_COMPLETE; events_update_idx=24; events_semaphore_value=1; id=2; semaphore=25 |

## DVE

| slot | opcode | BIR name | line | source | fields |
|---|---|---|---|---|---|
| 0 | `SET_ORDERING_MODE` | nan |  | `` | events_wait_mode=NONE; events_update_mode=NONE |
| 1 | `TENSOR_SCALAR_ARITH_OP` | I-7_inst__I-4-0 | 147 | `nisa.tensor_scalar(dst=acc, data=xg[:, 0, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=27; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=16; accumulator_cmd=IDLE; src_mem_pattern_start_addr_addr_immediate=218880; src_mem_pattern_step_elem=[1, 1, 1]; src_mem_pattern_num_elem=[960, 1, 1]; in_dtype=BFLOAT16; out_dtype=FP32; num_active_channels=128; imm0_src=POINTER_IMMEDIATE; op0=MULT; op1=BYPASS; reverse_operands=NONE; imm1_src=INSTRUCTION_IMMEDIATE; imm0_imm_arith_fp32=3.210655041461021e-40; dst_mem_pattern_start_addr_addr_immediate=20224; dst_mem_pattern_step_elem=[1, 1, 1]; dst_mem_pattern_ |
| 2 | `SCALAR_TENSOR_TENSOR_ARITH` | I-8_inst__I-4-0 | 150 | `nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=1; src0_mem_pattern_start_addr_addr_immediate=220800; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[960, 1]; src1_mem_pattern_start_addr_addr_immediate=20224; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[960, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=FP32; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE; dst_ |
| 3 | `TENSOR_SCALAR_ARITH_OP` | I-2_inst__I-4-0 | 147 | `nisa.tensor_scalar(dst=acc, data=xg[:, 0, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=28; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=16; accumulator_cmd=IDLE; src_mem_pattern_start_addr_addr_immediate=226560; src_mem_pattern_step_elem=[1, 1, 1]; src_mem_pattern_num_elem=[320, 1, 1]; in_dtype=BFLOAT16; out_dtype=FP32; num_active_channels=128; imm0_src=POINTER_IMMEDIATE; op0=MULT; op1=BYPASS; reverse_operands=NONE; imm1_src=INSTRUCTION_IMMEDIATE; imm0_imm_arith_fp32=3.210655041461021e-40; dst_mem_pattern_start_addr_addr_immediate=16384; dst_mem_pattern_step_elem=[1, 1, 1]; dst_mem_pattern_ |
| 4 | `SCALAR_TENSOR_TENSOR_ARITH` | I-9_inst__I-4-0 | 150 | `nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=2; src0_mem_pattern_start_addr_addr_immediate=222720; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[960, 1]; src1_mem_pattern_start_addr_addr_immediate=20224; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[960, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=FP32; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE; dst_ |
| 5 | `SCALAR_TENSOR_TENSOR_ARITH` | I-3_inst__I-4-0 | 150 | `nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=3; src0_mem_pattern_start_addr_addr_immediate=227200; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[320, 1]; src1_mem_pattern_start_addr_addr_immediate=16384; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[320, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=FP32; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE; dst_ |
| 6 | `SCALAR_TENSOR_TENSOR_ARITH` | I-10_inst__I-4-0 | 155 | `nisa.scalar_tensor_tensor(dst=yg[:, f0:f0 + w], data=xg[:, hc - 1, :],` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=NONE; events_semaphore_value=4; src0_mem_pattern_start_addr_addr_immediate=224640; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[960, 1]; src1_mem_pattern_start_addr_addr_immediate=20224; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[960, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=BFLOAT16; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE; dst_mem_pattern_start_addr_addr_i |
| 7 | `SCALAR_TENSOR_TENSOR_ARITH` | I-4_inst__I-4-0 | 150 | `nisa.scalar_tensor_tensor(dst=acc, data=xg[:, j, :], op0=nl.multiply,` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=5; src0_mem_pattern_start_addr_addr_immediate=227840; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[320, 1]; src1_mem_pattern_start_addr_addr_immediate=16384; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[320, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=FP32; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE; dst_ |
| 8 | `SCALAR_TENSOR_TENSOR_ARITH` | I-5_inst__I-4-0 | 155 | `nisa.scalar_tensor_tensor(dst=yg[:, f0:f0 + w], data=xg[:, hc - 1, :],` | events_wait_mode=WAIT_FOR_SEM_GE_IMM; events_wait_idx=6; events_update_mode=SEM_INC_COMPLETE; events_update_idx=6; events_semaphore_value=6; src0_mem_pattern_start_addr_addr_immediate=228480; src0_mem_pattern_step_elem=[1, 1]; src0_mem_pattern_num_elem=[320, 1]; src1_mem_pattern_start_addr_addr_immediate=16384; src1_mem_pattern_step_elem=[1, 1]; src1_mem_pattern_num_elem=[320, 1]; op0=MULT; op1=ADD; reverse_operands=NONE; imm0_src=POINTER_IMMEDIATE; in0_in1_dtype_dtype_lo=BFLOAT16; in0_in1_dtype_dtype_hi=FP32; out_dtype=BFLOAT16; num_active_channels=128; imm0_dtype=FP32; accumulator_cmd=IDLE;  |

## Activation

| slot | opcode | BIR name | line | source | fields |
|---|---|---|---|---|---|
| 0 | `SET_ORDERING_MODE` | nan |  | `` | events_wait_mode=NONE; events_update_mode=NONE |

## PE

| slot | opcode | BIR name | line | source | fields |
|---|---|---|---|---|---|
| 0 | `SET_ORDERING_MODE` | nan |  | `` | events_wait_mode=NONE; events_update_mode=NONE |

