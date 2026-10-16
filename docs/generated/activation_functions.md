# Activation function ids

Generated from the compiler's activation table sets (`neuronxcc/pwp/pwp_bin_trainium/*.json`, `profile_meta_data`). `ACTIVATE.activation_func` holds this id; the function must be in a table set the subgraph loads (chapter 3.5). The profiler's ACTIVATE names do not match these ids on Trainium2 (it prints `SOFTPLUS` for id 7, which is `exp`); trust this table. Verified against the NKI source for ids 2, 7, 29, 30, 31, 33, 36.

| id | function(s) | table sets that contain it |
|---|---|---|
| 1 | identity_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 2 | relu_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 3 | leaky_relu_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 4 | parametric_relu_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 5 | sigmoid_40p | sigmoid_and_others |
| 6 | tanh_4p | derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, sigmoid_and_others, silu_and_others, tanh_and_derivative |
| 7 | exp_400p | exp_and_others, natural_log_exp_and_others |
| 8 | sqrt_65536p | sqrt_and_others |
| 9 | softplus_40p | softplus_and_others |
| 10 | ln_400p, ln_40p | natural_log, natural_log_exp_and_others |
| 11 | derivative_identity_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 12 | derivative_relu_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 13 | derivative_leaky_relu_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 14 | derivative_sigmoid_40p | sigmoid_derivative |
| 15 | derivative_tanh_400p | tanh_and_derivative |
| 19 | sin_4p | derivative_silu_and_others, silu_and_others, trig_and_small |
| 21 | erf_4p | sigmoid_and_others |
| 22 | derivative_erf_400p | erf_derivative |
| 23 | gelu_4p | gelu_and_others |
| 24 | mish_4p | mish_and_others |
| 25 | gelu_apprx_tanh_40p | gelu_apprx_tanh_and_others |
| 26 | gelu_apprx_sigmoid_40p | gelu_apprx_sigmoid_and_others |
| 28 | arctan_4p | sigmoid_and_others, trig_and_small |
| 29 | reciprocal_sqrt_40000p | reciprocal_sqrt_and_small |
| 30 | square_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 31 | sign_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 32 | derivative_gelu_40p | gelu_and_others |
| 33 | abs_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 34 | abs_reciprocal_sqrt_40000p | abs_reciprocal_sqrt_and_small |
| 35 | is_finite_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 36 | silu_32p | silu_and_others |
| 37 | derivative_silu_32p | derivative_silu_and_others |
| 38 | derivative_gelu_apprx_sigmoid_4096p | derivative_gelu_apprx_sigmoid_and_others |
| 128 | copy_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
| 129 | reciprocal_400p | reciprocal_and_small |
| 130 | memset_zero_1p | abs_reciprocal_sqrt_and_small, derivative_gelu_apprx_sigmoid_and_others, derivative_silu_and_others, erf_derivative, exp_and_others, gelu_and_others, gelu_apprx_sigmoid_and_others, gelu_apprx_tanh_and_others, mish_and_others, natural_log, natural_log_exp_and_others, reciprocal_and_small, reciprocal_ |
