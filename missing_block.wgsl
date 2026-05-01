                    let assoc_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)));
                    let assoc_scale = 1.0 / max(assoc_rms, 0.1); // Soft normalization floor
                    let assoc_inj0 = alpha_assoc * assoc_ctx0 * assoc_scale;
                    let assoc_inj1 = alpha_assoc * assoc_ctx1 * assoc_scale;
                    let assoc_export0 = assoc_ctx0;
                    let assoc_export1 = assoc_ctx1;
                    AssocReadBuf[h_base_t + slot_offset + d0] = assoc_export0;
                    AssocReadBuf[h_base_t + slot_offset + d1] = assoc_export1;
                    fpm_ctx0 = fpm_ctx0 + assoc_inj0;
                    fpm_ctx1 = fpm_ctx1 + assoc_inj1;
                    if (ENABLE_ASSOC_POST_HSTAR) {
                        assoc_post0 = assoc_inj0;
                        assoc_post1 = assoc_inj1;
                    }
                }
                workgroupBarrier();
            }

            var local_sumsq = 0.0;
            if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
                // h_hist is stop-grad (∂/∂h=0) → excluded from rms denominator
                // same principle as hist_ctx exclusion
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let attn0 = select(0.0, Scratch[attn_base + d0] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let attn1 = select(0.0, Scratch[attn_base + d1] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1];
                H_next[h_base_t + slot_offset + d0] = h_dep0 + hhist_gamma_wg * h_hist0;
                H_next[h_base_t + slot_offset + d1] = h_dep1 + hhist_gamma_wg * h_hist1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1; // rms from h-dep only
            } else if (fpm_inject_enabled && d_model == WG_SIZE * 2u) {
                // Model A: memory contributes a token-fixed context inside the H-only solve.
                // Like h_hist, it is context from the past, not a fresh H-dependent branch,
                // so it should not inflate the RMS denominator of the H operator itself.
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let attn0 = select(
                    0.0,
                    Scratch[attn_base + d0] * slot_attn_scale,
                    recurrent_slot_attn_enabled || iter == 0u,
                );
                let attn1 = select(
                    0.0,
                    Scratch[attn_base + d1] * slot_attn_scale,
                    recurrent_slot_attn_enabled || iter == 0u,
                );
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1];
                let pre0 = h_dep0 + fpm_ctx0;
                let pre1 = h_dep1 + fpm_ctx1;
                H_next[h_base_t + slot_offset + d0] = pre0;
                H_next[h_base_t + slot_offset + d1] = pre1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
            } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let attn0 = select(0.0, Scratch[attn_base + d0] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let attn1 = select(0.0, Scratch[attn_base + d1] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0] + hist_mem0;
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1] + hist_mem1;
                H_next[h_base_t + slot_offset + d0] = h_dep0;
                H_next[h_base_t + slot_offset + d1] = h_dep1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
            } else {
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + slot_offset + d];
                    let attn = select(0.0, Scratch[attn_base + d] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                    let assoc_inj_d = select(assoc_inj0, assoc_inj1, d >= WG_SIZE);
                    let pre = Scratch[signal_base + d]
                        + H_SELF_FEEDBACK * h_prev
                        + attn
                        + AllWeights[slot_anchor_mat_base + d]
                        + assoc_inj_d;
                    H_next[h_base_t + slot_offset + d] = pre;
                    local_sumsq = local_sumsq + pre * pre;
                }
            }
            shared_vals[tid] = local_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);

            let alpha_h = select(FPM_ALPHA_H, FPM_ALPHA_H * 0.5, rescue_active);
            var local_delta_h_num = 0.0;
            var local_delta_h_den = 0.0;
            var local_fh_sq = 0.0;
            var local_nscale_abs = 0.0;
            var local_attn_sq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_offset + d];
                let nscale = AllWeights[nscale_base + d];
                let attn_val = Scratch[attn_base + d] * slot_attn_scale;
                let f_h = nscale * (H_next[h_base_t + slot_offset + d] / rms);
                let blend = select(shape.damping, alpha_h, fpm_policy_enabled && d_model == WG_SIZE * 2u);
                let val = blend * f_h + (1.0 - blend) * h_prev;
                let delta_h = val - h_prev;
                local_delta_h_num = local_delta_h_num + delta_h * delta_h;
                local_delta_h_den = local_delta_h_den + h_prev * h_prev;
                local_fh_sq = local_fh_sq + f_h * f_h;
                local_nscale_abs = max(local_nscale_abs, abs(nscale));
                local_attn_sq = local_attn_sq + attn_val * attn_val;
                H_curr[h_base + slot_offset + d] = val;
                H_next[h_base_t + slot_offset + d] = val;
            }
            shared_vals[tid] = local_delta_h_num;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_h_num = shared_vals[0];
            shared_vals[tid] = local_delta_h_den;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let hprev_energy = shared_vals[0];
            let delta_h_rms = sqrt(delta_h_num * inv_d_model + 1e-12);
            let err_h_operator = delta_h_rms / max(rms, 1e-6);
            let err_h = err_h_operator;
            let hprev_rms = sqrt(hprev_energy * inv_d_model + 1e-6);
            shared_vals[tid] = local_fh_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let fh_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            shared_vals[tid] = local_attn_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let attn_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            shared_vals[tid] = local_nscale_abs;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let nscale_abs = shared_vals[0];
            var local_max_delta_a = 0.0;
            if (fpm_inject_enabled && tid < h_slots) {
                local_max_delta_a = abs(slot_coord_prev[tid] - slot_coord_weights[tid]);
            }
            shared_vals[tid] = local_max_delta_a;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_a = shared_vals[0];
            // FPM plastic write deferred to once per token after convergence (below).
            // fpm_ctx reads the previous token's frozen memory snapshot:
            // HistCtx[t-1] within the chunk, MState only for the first absolute token.
            // That preserves Picard stability while restoring causal token-to-token memory.
            var local_max_delta_m = 0.0;
            var local_gate = 0.0;
            var local_update_ratio = 0.0;
            var local_delta_m_num = 0.0;
            var local_delta_m_den = 0.0;
            shared_vals[tid] = local_delta_m_num;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_num = shared_vals[0];
            shared_vals[tid] = local_delta_m_den;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let err_m = sqrt(delta_m_num / (shared_vals[0] + FPM_EPS));
            if (err_h == err_h && abs(err_h) < 1e30) {
                last_finite_err_h = err_h;
            }
            shared_vals[tid] = local_max_delta_m;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_m = shared_vals[0];
            shared_vals[tid] = err_h;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }

            let stop_err_h = shared_vals[0];
            let homeo_band = max(shape.epsilon, FPM_HOMEO_ALPHA_ERR_SCALE * alpha_h);
            let plateau_ratio = abs(stop_err_h - prev_err_h) / max(prev_err_h, shape.epsilon);
            let strict_converged = stop_err_h < shape.epsilon;
            let homeostatic_converged =
                iter + 1u >= FPM_HOMEO_MIN_ITERS
                && stop_err_h <= homeo_band
                && plateau_ratio <= FPM_HOMEO_PLATEAU_TOL;
            if (tid == 0u && (strict_converged || homeostatic_converged)) {
                converged_flag_wg = 1u;
                if (strict_converged) {
                    token_strict_converged = true;
                } else {
                    token_homeostatic_converged = true;
                }
                if (!rescue_active) {
                    converged_before_rescue = true;
                }
            }
            workgroupBarrier();
            converged = converged_flag_wg != 0u;
            if (tid == 0u) {
                let d_curr = stop_err_h;
                let d_prev = last_delta;
                var recurrent_iter = iter;
                if ((enable_slot_coord && !ENABLE_FPM || model_a_memory_bootstrap) && iter > 0u) {
                    recurrent_iter = iter - 1u;
                }
                if (recurrent_iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    max_contractivity = max(max_contractivity, d_curr / d_prev);
                }
                last_delta = d_curr;
                if (!((enable_slot_coord && !ENABLE_FPM || model_a_memory_bootstrap) && iter == 0u)) {
                    max_delta_seen = max(max_delta_seen, d_curr);
                }
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m);
                max_a_delta_seen = max(max_a_delta_seen, iter_max_delta_a);
                if (stop_err_h > max_err_h_marker_seen) {
                    max_err_h_marker_seen = stop_err_h;
                    iter_of_max_err_h_seen = f32(iter);
                    token_of_max_err_h_seen = f32(t);
                }
                let attn_to_signal = attn_rms / max(solve_signal_rms_seen, 1e-6);
                if (attn_to_signal > max_attn_ratio_marker_seen) {
                    max_attn_ratio_marker_seen = attn_to_signal;
                    iter_of_max_attn_ratio_seen = f32(iter);
                    token_of_max_attn_ratio_seen = f32(t);
                }
                max_err_h_seen = max(max_err_h_seen, stop_err_h);
                max_err_m_seen = max(max_err_m_seen, err_m);
                max_z_seen = max(max_z_seen, local_gate);
                max_update_ratio_seen = max(max_update_ratio_seen, local_update_ratio);
                solve_pre_rms_seen = max(solve_pre_rms_seen, rms);
                solve_fh_rms_seen = max(solve_fh_rms_seen, fh_rms);
                solve_hprev_rms_seen = max(solve_hprev_rms_seen, hprev_rms);
                solve_nscale_abs_seen = max(solve_nscale_abs_seen, nscale_abs);
                solve_pre_to_hprev_seen =
                    max(solve_pre_to_hprev_seen, rms / max(hprev_rms, 1e-6));
                solve_fh_to_hprev_seen =
                    max(solve_fh_to_hprev_seen, fh_rms / max(hprev_rms, 1e-6));
                solve_attn_rms_seen = max(solve_attn_rms_seen, attn_rms);
                solve_attn_to_signal_seen = max(solve_attn_to_signal_seen, attn_to_signal);
                solve_attn_scale_seen = max(solve_attn_scale_seen, abs(slot_attn_scale));
                max_homeo_band_seen = max(max_homeo_band_seen, homeo_band);
                if (iter == 0u) {
                    iter0_err_h_seen = max(iter0_err_h_seen, stop_err_h);
                    iter0_attn_to_signal_seen = max(
                        iter0_attn_to_signal_seen,
                        attn_rms / max(solve_signal_rms_seen, 1e-6),
                    );
                    iter0_attn_scale_seen = max(iter0_attn_scale_seen, abs(slot_attn_scale));
                } else if (iter == 1u) {
                    iter1_err_h_seen = max(iter1_err_h_seen, stop_err_h);
                    iter1_attn_to_signal_seen = max(
                        iter1_attn_to_signal_seen,
                        attn_rms / max(solve_signal_rms_seen, 1e-6),
                    );
                    iter1_attn_scale_seen = max(iter1_attn_scale_seen, abs(slot_attn_scale));
                }
                token_max_m_delta = max(token_max_m_delta, iter_max_delta_m);
                token_max_a_delta = max(token_max_a_delta, iter_max_delta_a);
                if (rescue_active && !converged && iter == iter_limit) {
                    rescue_count_seen = rescue_count_seen + 1.0;
                }
            }
            prev_err_h = stop_err_h;
            if (converged && rescue_triggered && tid == 0u) {
                rescue_recovered = true;
            }
            iter = iter + 1u;
            workgroupBarrier();
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 181.0;
        }
        if (tid == 0u) {
            total_iters_seen = total_iters_seen + iter;
            sum_slot_move_seen = sum_slot_move_seen + token_max_m_delta;
            exit_iter_sum = exit_iter_sum + f32(iter);
            let prev_err_h_finite = prev_err_h == prev_err_h && abs(prev_err_h) < 1e30;
            let exit_err_h = select(prev_err_h, last_finite_err_h, !prev_err_h_finite);
            if (exit_err_h == exit_err_h && abs(exit_err_h) < 1e30) {
                exit_err_h_sum = exit_err_h_sum + exit_err_h;
                exit_err_h_valid_sum = exit_err_h_valid_sum + 1.0;
            }
            var entropy = 0.0;
            for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                let w = slot_coord_weights[ms];
                entropy = entropy - w * log(max(w, 1e-12));
            }
            sum_self_assign_seen = sum_self_assign_seen + slot_coord_weights[slot_idx];
            sum_assign_entropy_seen = sum_assign_entropy_seen + entropy;
            if (slot_coord_weights[slot_idx] < FPM_DEAD_THRESHOLD) {
                dead_slot_seen = dead_slot_seen + 1.0;
            }
            if (max_z_seen > FPM_SAT_THRESHOLD) {
                write_saturation_seen = write_saturation_seen + 1.0;
            }
            if (rescue_recovered) {
                rescue_recovered_seen = rescue_recovered_seen + 1.0;
            }
            if (converged_before_rescue) {
                pre_rescue_converged_sum = pre_rescue_converged_sum + 1.0;
            }
            if (token_strict_converged) {
                strict_converged_sum = strict_converged_sum + 1.0;
            } else if (token_homeostatic_converged) {
                homeostatic_converged_sum = homeostatic_converged_sum + 1.0;
            } else {
                failed_converged_sum = failed_converged_sum + 1.0;
            }
            if (!converged) {
                failed_hits_seen = failed_hits_seen + 1u;
            }
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 182.0;
        }
        workgroupBarrier();

        if (ENABLE_ASSOC_POST_HSTAR && assoc_read_enabled && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            H_curr[h_base + slot_offset + d0] = H_curr[h_base + slot_offset + d0] + assoc_post0;
            H_curr[h_base + slot_offset + d1] = H_curr[h_base + slot_offset + d1] + assoc_post1;
            H_next[h_base_t + slot_offset + d0] = H_curr[h_base + slot_offset + d0];
            H_next[h_base_t + slot_offset + d1] = H_curr[h_base + slot_offset + d1];
        }
        workgroupBarrier();

        // h_currSSM: h_ssm = a * h_ssm + (1-a) * h*
        //   a[d] = sigmoid(-A_log[d]) — per-dim time constant, zero extra params
        //   state lives directly in h* space, no projections needed
        if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let alog_off = aw_alog_base(d_model, h_slots) + slot_offset;
            let a0 = 1.0 / (1.0 + exp(AllWeights[alog_off + d0]));  // sigmoid(-A_log)
            let a1 = 1.0 / (1.0 + exp(AllWeights[alog_off + d1]));
            let h_star0 = H_curr[h_base + slot_offset + d0];
            let h_star1 = H_curr[h_base + slot_offset + d1];
            H_hist[h_base + slot_offset + d0] = a0 * h_hist0 + (1.0 - a0) * h_star0;
            H_hist[h_base + slot_offset + d1] = a1 * h_hist1 + (1.0 - a1) * h_star1;
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 183.0;
        }
        workgroupBarrier();

        // FPM plastic write: once per token using h* (converged fixed point).
        // Model A memory should be a temporal carrier, not just a gated proposal buffer.
        // The base state therefore follows the same slot-wise temporal dynamics as the CPU
        // reference path (a_log / w_x / w_out), while retain/z/proposal modulate the novelty
        // injected into that carrier.
        var assoc_write_gate_token = 1.0;
        if (fpm_write_enabled && !is_segment_memory_slot && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let h_val0 = H_curr[h_base + slot_offset + d0];
            let h_val1 = H_curr[h_base + slot_offset + d1];
            let c0 = 0.5 * (h_val0 + Scratch[signal_base + d0]);
            let c1 = 0.5 * (h_val1 + Scratch[signal_base + d1]);
            let m_prev0 = fpm_m_cache[d0];
            let m_prev1 = fpm_m_cache[d1];
            let alpha_m = clamp(shape.fpm_alpha_m, 0.01, 0.1);
            let residual_scale = FPM_RESIDUAL_SCALE;
            let hist_base = aw_hist_base(d_model, h_slots);
            let wx_base = aw_wx_base(d_model, h_slots);
            let wout_base = aw_wout_base(d_model, h_slots);
            let alog_off = aw_alog_base(d_model, h_slots) + slot_offset;
            let h_sq = h_val0 * h_val0 + h_val1 * h_val1;
            shared_vals[tid] = h_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let h_rms = sqrt(max(shared_vals[0] / max(1.0, f32(d_model)), 1e-6));
            let h_unit0 = h_val0 / h_rms;
            let h_unit1 = h_val1 / h_rms;
            let wf_base = hist_base + w_write_gate_base(d_model, h_slots) + slot_idx * d_model;
            var gate_partial = 0.0;
            for (var j = tid; j < d_model; j = j + WG_SIZE) {
                let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                gate_partial = gate_partial + AllWeights[wf_base + j] * c_j;
            }
            shared_vals[tid] = gate_partial;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let gate_bias = AllWeights[hist_base + b_write_mem_base(d_model, h_slots) + slot_idx]
                + FPM_GATE_BIAS;
            let raw_z = 1.0 / (1.0 + exp(-(shared_vals[0] * inverseSqrt(max(1.0, f32(d_model))) + gate_bias)));
            // Factored k×v write proposal: bottleneck = W_k_write·c  (d→r), proposal = tanh(W_v_write·bottleneck + b)
            let wkw_base = hist_base + w_k_write_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
            let wvw_base = hist_base + w_v_write_base(d_model, h_slots) + slot_idx * RETAIN_RANK * d_model;
            let bd_base  = hist_base + hist_delta_bias_base(d_model, h_slots) + slot_idx * d_model;
            // Step 1: bottleneck[r] = Σ_j W_k_write[j,r] * c_j  (RETAIN_RANK lanes)
            if (tid < RETAIN_RANK) {
                var kw_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                    kw_acc = kw_acc + AllWeights[wkw_base + j * RETAIN_RANK + tid] * c_j;
                }
                shared_vals[tid] = kw_acc;
            }
            workgroupBarrier();
            // Step 2: proposal[d] = tanh(Σ_r W_v_write[r,d] * bottleneck[r] + b_delta[d])
            var delta_in0 = AllWeights[bd_base + d0];
            var delta_in1 = AllWeights[bd_base + d1];
            for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                delta_in0 = delta_in0 + AllWeights[wvw_base + r * d_model + d0] * shared_vals[r];
                delta_in1 = delta_in1 + AllWeights[wvw_base + r * d_model + d1] * shared_vals[r];
            }
            let proposal0 = tanh(delta_in0);
            let proposal1 = tanh(delta_in1);
            let proposal_sq = proposal0 * proposal0 + proposal1 * proposal1;
            shared_vals[tid] = proposal_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let proposal_norm = sqrt(max(shared_vals[0], 1e-6));
            let prev_sq = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            shared_vals[tid] = prev_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let prev_norm = sqrt(max(shared_vals[0], 1e-6));
            let diff0 = proposal0 - m_prev0;
            let diff1 = proposal1 - m_prev1;
            shared_vals[tid] = diff0 * diff0 + diff1 * diff1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            // Structural novelty measures actual disagreement between the candidate write
            // and the carried memory, not just the magnitude of the candidate.
            let diff_norm = sqrt(max(shared_vals[0], 1e-6));
            let novelty = diff_norm / (diff_norm + prev_norm + 1e-6);
            let z = clamp(raw_z, 0.0, 1.0);
            // Retain gate (low-rank r=32): retain = σ(W_down · (W_up · c) + b_retain)
            // Replaces uniform fatigue decay with input-dependent selective forgetting.
            let hist_base_r = aw_hist_base(d_model, h_slots);
            let wup_base = hist_base_r + w_retain_up_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
            let wdown_base = hist_base_r + w_retain_down_base(d_model, h_slots) + slot_idx * RETAIN_RANK * d_model;
            let bret_base = hist_base_r + b_retain_base(d_model, h_slots) + slot_idx * d_model;
            // Step 1: up = W_up · c  (d_model → RETAIN_RANK)
            // Threads tid < RETAIN_RANK each compute one element of up, store in shared_vals[0..32]
            if (tid < RETAIN_RANK) {
                var up_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                    up_acc = up_acc + AllWeights[wup_base + j * RETAIN_RANK + tid] * c_j;
                }
                shared_vals[tid] = up_acc;
            }
            workgroupBarrier();
            // Step 2: retain[d] = σ(W_down[d,:] · up + b_retain[d])
            var down_acc0 = AllWeights[bret_base + d0];
            var down_acc1 = AllWeights[bret_base + d1];
            for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                down_acc0 = down_acc0 + AllWeights[wdown_base + r * d_model + d0] * shared_vals[r];
                down_acc1 = down_acc1 + AllWeights[wdown_base + r * d_model + d1] * shared_vals[r];
            }
            let retain_raw0 = 1.0 / (1.0 + exp(-down_acc0));
            let retain_raw1 = 1.0 / (1.0 + exp(-down_acc1));
            // Retain is now a true preservation gate:
            // when novelty is low, preserve memory regardless of local write preference;
            // when novelty is high, the learned retain gate decides which dimensions stay fixed.
            let retain0 = 1.0 - novelty * (1.0 - retain_raw0);
            let retain1 = 1.0 - novelty * (1.0 - retain_raw1);
            let write_budget0 = (1.0 - retain0) * z;
            let write_budget1 = (1.0 - retain1) * z;
            let write0 = z * (residual_scale * proposal0);
            let write1 = z * (residual_scale * proposal1);
            let wx0 = 0.5 * tanh(AllWeights[wx_base + d0 * d_model + d0]);
            let wx1 = 0.5 * tanh(AllWeights[wx_base + d1 * d_model + d1]);
            let wx_term0 = wx0 * h_unit0;
            let wx_term1 = wx1 * h_unit1;
            let h_write0 = sqrt(max(z, 1.0e-6)) * h_unit0;
            let h_write1 = sqrt(max(z, 1.0e-6)) * h_unit1;
            let x_proj0 = h_write0 + wx_term0 + write0;
            let x_proj1 = h_write1 + wx_term1 + write1;
            let a0 = 1.0 / (1.0 + exp(AllWeights[alog_off + d0]));
            let a1 = 1.0 / (1.0 + exp(AllWeights[alog_off + d1]));
            let base_inner0 = a0 * m_prev0 + (1.0 - a0) * x_proj0;
            let base_inner1 = a1 * m_prev1 + (1.0 - a1) * x_proj1;
            let m_inner0 = retain0 * m_prev0 + (1.0 - retain0) * base_inner0;
            let m_inner1 = retain1 * m_prev1 + (1.0 - retain1) * base_inner1;
            H_next[h_base_t + slot_offset + d0] = m_inner0;
            H_next[h_base_t + slot_offset + d1] = m_inner1;
            workgroupBarrier();
            var out_acc0 = m_inner0;
            var out_acc1 = m_inner1;
            for (var j = 0u; j < d_model; j = j + 1u) {
                let m_inner_j = H_next[h_base_t + slot_offset + j];
                out_acc0 = out_acc0 + AllWeights[wout_base + d0 * d_model + j] * m_inner_j;
                out_acc1 = out_acc1 + AllWeights[wout_base + d1 * d_model + j] * m_inner_j;
            }
            // W_out remains observable as a readout/refinement term, but the recurrent
            // FPM state itself is m_inner. Recirculating (I + W_out)m_inner as memory made
            // the state semantics depend on a readout transform and became unstable when
            // FPM was given enough alpha to matter.
            let m_candidate0 = out_acc0;
            let m_candidate1 = out_acc1;
            let wout_term0 = m_candidate0 - m_inner0;
            let wout_term1 = m_candidate1 - m_inner1;
            let proposal_rms_p = sqrt(0.5 * (proposal0 * proposal0 + proposal1 * proposal1));
            let candidate_rms_p = sqrt(0.5 * (m_candidate0 * m_candidate0 + m_candidate1 * m_candidate1));
            let retain_avg_p = 0.5 * (retain0 + retain1);
            let retain_max_p = max(retain0, retain1);
            if (debug_on) {
                shared_vals[tid] = h_write0 * h_write0 + h_write1 * h_write1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let h_write_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = wx_term0 * wx_term0 + wx_term1 * wx_term1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let wx_term_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = write0 * write0 + write1 * write1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let write_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = m_inner0 * m_inner0 + m_inner1 * m_inner1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let m_inner_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = wout_term0 * wout_term0 + wout_term1 * wout_term1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let wout_term_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                if (tid == 0u) {
                    let slot_write_base = 640u + slot_idx * 6u;
                    DebugLog[slot_write_base + 0u] = max(DebugLog[slot_write_base + 0u], h_write_rms_p);
                    DebugLog[slot_write_base + 1u] = max(DebugLog[slot_write_base + 1u], wx_term_rms_p);
                    DebugLog[slot_write_base + 2u] = max(DebugLog[slot_write_base + 2u], write_rms_p);
                    DebugLog[slot_write_base + 3u] = max(DebugLog[slot_write_base + 3u], m_inner_rms_p);
                    DebugLog[slot_write_base + 4u] = max(DebugLog[slot_write_base + 4u], wout_term_rms_p);
                    DebugLog[slot_write_base + 5u] = max(DebugLog[slot_write_base + 5u], candidate_rms_p);
                }
                workgroupBarrier();
            }
            // Diagnostics: replicate the same reductions as the original per-iter block.
            var local_delta_m_num_p = (m_inner0 - m_prev0) * (m_inner0 - m_prev0)
                                    + (m_inner1 - m_prev1) * (m_inner1 - m_prev1);
            shared_vals[tid] = local_delta_m_num_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_num_p = shared_vals[0];
            var local_delta_m_prev_den_p = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            shared_vals[tid] = local_delta_m_prev_den_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_prev_den_p = shared_vals[0];
            let local_delta_m_cand_den_p = m_inner0 * m_inner0 + m_inner1 * m_inner1;
            shared_vals[tid] = local_delta_m_cand_den_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_cand_den_p = shared_vals[0];
            let err_m_p = sqrt(delta_m_num_p / max(max(delta_m_prev_den_p, delta_m_cand_den_p), FPM_EPS));
            let local_max_delta_m_p = max(abs(m_inner0 - m_prev0), abs(m_inner1 - m_prev1));
            shared_vals[tid] = local_max_delta_m_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_m_p = shared_vals[0];
            let upd_num_p = write0 * write0 + write1 * write1;
            let upd_den_p = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            let update_ratio_p = sqrt(upd_num_p / (upd_den_p + FPM_EPS));
            let local_assoc_write_budget = write_budget0 + write_budget1;
            shared_vals[tid] = local_assoc_write_budget;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            assoc_write_gate_token = ASSOC_WRITE_CAP * raw_z;
            if (tid == 0u) {
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m_p);
                max_err_m_seen = max(max_err_m_seen, err_m_p);
                max_z_seen = max(max_z_seen, z);
                max_update_ratio_seen = max(max_update_ratio_seen, update_ratio_p);
                sum_slot_move_seen = sum_slot_move_seen + iter_max_delta_m_p;
                if (z > FPM_SAT_THRESHOLD) {
                    write_saturation_seen = write_saturation_seen + 1.0;
                }
                let slot_probe_base = 520u + slot_idx * 14u;
                DebugLog[slot_probe_base + 2u] = max(DebugLog[slot_probe_base + 2u], retain_max_p);
                DebugLog[slot_probe_base + 3u] = DebugLog[slot_probe_base + 3u] + retain_avg_p;
                DebugLog[slot_probe_base + 4u] = max(DebugLog[slot_probe_base + 4u], proposal_rms_p);
                DebugLog[slot_probe_base + 5u] = max(DebugLog[slot_probe_base + 5u], candidate_rms_p);
            }
            fpm_m_cache[d0] = m_inner0;
            fpm_m_cache[d1] = m_inner1;
            workgroupBarrier();
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 184.0;
        }
        workgroupBarrier();

        // Model A token-local history: once write is enabled, each token materializes its
        // internal FPM state into HistCtx so the next token in the same chunk can read it.
        // Inter-chunk persistence remains a separate stage handled by the bridge via MState.
        if (d_model == WG_SIZE * 2u) {
            let working0 = fpm_m_cache[tid];
            let working1 = fpm_m_cache[tid + WG_SIZE];
            // Per-token storage for the causal history snapshot and retain-gate backward.
            // The bridge Rust-side copies HistCtx[last_token] → MState after each chunk (stage>=4).
            HistCtx[h_base_t + slot_offset + tid] = working0;
            HistCtx[h_base_t + slot_offset + tid + WG_SIZE] = working1;
        }
        if (assoc_write_enabled && !is_segment_memory_slot && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let hist_base = aw_hist_base(d_model, h_slots);
            // Associative write keys use their own address encoder. Sharing with the
            // FPM write-key made a single unstable associative run corrupt the core
            // FPM write geometry and bifurcate training.
            let wk_assoc = hist_base + w_k_assoc_base(d_model, h_slots) + slot_idx * d_model * ASSOC_RANK;
            // W_v_assoc is reserved for the transition-gate auxiliary branch. The default
            // bank-value path stores raw token identity directly into bank_value so the
            // explicit associative memory keeps token semantics without a hidden projector.
            let wv_assoc = hist_base + w_v_assoc_base(d_model, h_slots) + slot_idx * d_model * ASSOC_RANK;
            let wevent_assoc = hist_base + w_event_assoc_base(d_model, h_slots) + slot_offset;
            let bevent_assoc = hist_base + b_event_assoc_base(d_model, h_slots) + slot_idx;
            let assoc_anchor_base = hist_base + slot_anchor_base(d_model, h_slots) + slot_offset;
            // Layout per bank: [bank_key | bank_value | usage].
            let assoc_hist_base =
                ((batch_idx * shape.token_count + t) * h_slots + slot_idx) * assoc_hist_slot_stride;
            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                let hist_bank_base = assoc_hist_base + bank * assoc_bank_stride;
                if (tid < ASSOC_RANK) {
                    AssocHist[hist_bank_base + tid] = AssocBuf[bank_base + tid];
                }
                AssocHist[hist_bank_base + ASSOC_RANK + d0] =
                    AssocBuf[bank_base + ASSOC_RANK + d0];
                AssocHist[hist_bank_base + ASSOC_RANK + d1] =
                    AssocBuf[bank_base + ASSOC_RANK + d1];
                if (tid == 0u) {
                    AssocHist[hist_bank_base + ASSOC_RANK + d_model] =
                        AssocBuf[bank_base + ASSOC_RANK + d_model];
                }
            }
            workgroupBarrier();
            let assoc_raw0 =
                Scratch[signal_base + d0]
                + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
            let assoc_raw1 =
                Scratch[signal_base + d1]
                + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            shared_vals[tid] = assoc_raw0 * assoc_raw0 + assoc_raw1 * assoc_raw1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let assoc_signal_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            let assoc_src0 = assoc_raw0 / max(assoc_signal_rms, 1.0e-6);
            let assoc_src1 = assoc_raw1 / max(assoc_signal_rms, 1.0e-6);
            let has_prev_hstar = select(0.0, 1.0, t > 0u);
            var event_prev0 = 0.0;
            var event_prev1 = 0.0;
            var event_curr0 = 0.0;
            var event_curr1 = 0.0;
            if (t > 0u) {
                let prev_s_in_base = (batch_idx * shape.seq_len + (global_t - 1u)) * d_model;
                event_prev0 =
                    S_in[prev_s_in_base + d0]
                    + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
                event_prev1 =
                    S_in[prev_s_in_base + d1]
                    + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            }
            event_curr0 =
                S_in[s_in_base + d0]
                + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
            event_curr1 =
                S_in[s_in_base + d1]
                + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            shared_vals[tid] = event_prev0 * event_prev0 + event_prev1 * event_prev1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let event_prev_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            shared_vals[tid] = event_curr0 * event_curr0 + event_curr1 * event_curr1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let event_curr_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            // Event gate is a learned transition classifier over prev_source → curr_source.
            // The gate must react to change, not just signal magnitude; otherwise it becomes
            // an almost-constant write valve that opens for nearly every token pair.
            shared_vals[tid] =
                AllWeights[wevent_assoc + d0]
                    * (
                        (
                            event_curr0 / max(event_curr_rms, 1.0e-6)
                            - event_prev0 / max(event_prev_rms, 1.0e-6)
                        )
                        + (
                            event_curr0 / max(event_curr_rms, 1.0e-6)
                            * event_prev0 / max(event_prev_rms, 1.0e-6)
                        )
                    )
                + AllWeights[wevent_assoc + d1]
                    * (
                        (
                            event_curr1 / max(event_curr_rms, 1.0e-6)
                            - event_prev1 / max(event_prev_rms, 1.0e-6)
                        )
                        + (
                            event_curr1 / max(event_curr_rms, 1.0e-6)
                            * event_prev1 / max(event_prev_rms, 1.0e-6)
                        )
                    );
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let learned_event_gate = 1.0 / (
                1.0 + exp(-(shared_vals[0] * inv_sqrt_d_model + AllWeights[bevent_assoc]))
            );
            let event_gate = select(1.0, learned_event_gate, ENABLE_ASSOC_EVENT_GATE);
            // Structural split:
            // - the explicit associative bank writes when the transition classifier says
            //   "this pair is a binding worth storing";
            // - the FPM write budget only controls how much of that selected binding is
            //   consolidated into the slower recurrent state.
            // Using the FPM gate for both roles kept the system stable but starved binding.
            let bind_gate = event_gate * has_prev_hstar;
            // Durable binding update:
            //   bank_key   <- (1-g) bank_key   + g * W_k(source_{t-1})
            //   bank_value <- (1-g) bank_value + g * token_identity_t
            // Keys stay in the explicit associative source space so token identity remains
            // sharp. Values are written as the normalized raw current token identity: for
            // associative recall the thing we want back is the observed VAL token itself,
            // not a representation already mixed by the solve.
            var assoc_value0 = 0.0;
            var assoc_value1 = 0.0;
            if (tid < ASSOC_RANK) {
                var prev_sig_sq = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_sig_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sig_sq = prev_sig_sq + prev_sig_j * prev_sig_j;
                }
                let prev_sig_rms = sqrt(prev_sig_sq / max(1.0, f32(d_model)) + 1.0e-6);
                var k_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_sig_j = PrevHStarBuf[prev_hstar_base + j] / max(prev_sig_rms, 1.0e-6); // associative source at t-1
                    k_acc = k_acc + AllWeights[wk_assoc + j * ASSOC_RANK + tid] * prev_sig_j;
                }
                // Reuse attention-local workgroup scratch in the post-solve assoc write path.
                // At this point of the token loop q_self/head_mix are dead, so they can safely
                // preserve the key/value code until allocator + bank write finish.
                q_self[tid] = tanh(k_acc);
                head_mix[tid] = 0.0;
                if (ENABLE_ASSOC_TRANSITION_GATE) {
                    var v_acc = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let curr_token_j = S_in[s_in_base + j] / max(event_curr_rms, 1.0e-6);
                        v_acc = v_acc + AllWeights[wv_assoc + j * ASSOC_RANK + tid] * curr_token_j;
                    }
                    head_mix[tid] = tanh(
                        v_acc * inverseSqrt(max(1.0, f32(ASSOC_RANK)))
                    );
                }
            }
            workgroupBarrier();
            assoc_value0 = S_in[s_in_base + d0] / max(event_curr_rms, 1.0e-6);
            assoc_value1 = S_in[s_in_base + d1] / max(event_curr_rms, 1.0e-6);
            Scratch[signal_base + d0] = assoc_value0;
            Scratch[signal_base + d1] = assoc_value1;
            shared_vals[tid] = assoc_value0 * assoc_value0 + assoc_value1 * assoc_value1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let assoc_value_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            assoc_value0 = Scratch[signal_base + d0] / max(assoc_value_rms, 1.0e-6);
            assoc_value1 = Scratch[signal_base + d1] / max(assoc_value_rms, 1.0e-6);
            if (tid == 0u && debug_on) {
                var key_sq = 0.0;
                var prev_sq = 0.0;
                var kpre_sq = 0.0;
                for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                    let kv = q_self[r];
                    key_sq = key_sq + kv * kv;
                    let kpre = atanh(clamp(kv, -0.999999, 0.999999));
                    kpre_sq = kpre_sq + kpre * kpre;
                }
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sq = prev_sq + prev_j * prev_j;
                }
                let assoc_write_probe_base = 900u + slot_idx * 6u;
                DebugLog[assoc_write_probe_base + 0u] =
                    DebugLog[assoc_write_probe_base + 0u] + sqrt(key_sq / max(1.0, f32(ASSOC_RANK)));
                DebugLog[assoc_write_probe_base + 1u] =
                    DebugLog[assoc_write_probe_base + 1u] + assoc_value_rms;
                DebugLog[assoc_write_probe_base + 2u] = DebugLog[assoc_write_probe_base + 2u] + 1.0;
                DebugLog[assoc_write_probe_base + 3u] =
                    DebugLog[assoc_write_probe_base + 3u] + sqrt(prev_sq / max(1.0, f32(d_model)));
                DebugLog[assoc_write_probe_base + 4u] =
                    DebugLog[assoc_write_probe_base + 4u] + sqrt(kpre_sq / max(1.0, f32(ASSOC_RANK)));
                DebugLog[assoc_write_probe_base + 5u] = DebugLog[assoc_write_probe_base + 5u];
            }
            if (tid == 0u) {
                var chosen_bank = 0u;
                var found_empty = false;
                var empty_bank = 0u;
                var best_cos = -1.0e30;
                var best_value_cos = -1.0e30;
                var best_bank = 0u;
                var min_usage = 1.0e30;
                var min_usage_bank = 0u;
                var allow_write = 0.0;
                var overwrite_bank = 0.0;
                var new_key_norm = 0.0;
                var new_value_norm = 0.0;
                for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                    new_key_norm = new_key_norm + q_self[r] * q_self[r];
                }
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let v_j = Scratch[signal_base + j] / max(assoc_value_rms, 1.0e-6);
                    new_value_norm = new_value_norm + v_j * v_j;
                }
                var bank_scores: array<f32, 8>; 
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                    var key_norm = 0.0;
                    var score = 0.0;
                    var value_norm = 0.0;
                    var value_score = 0.0;
                    for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                        let key_r = AssocBuf[bank_base + r];
                        key_norm = key_norm + key_r * key_r;
                        score = score + key_r * q_self[r];
                    }
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let bank_v = AssocBuf[bank_base + ASSOC_RANK + j];
                        let curr_v = Scratch[signal_base + j] / max(assoc_value_rms, 1.0e-6);
                        value_norm = value_norm + bank_v * bank_v;
                        value_score = value_score + bank_v * curr_v;
                    }
                    let bank_usage = AssocBuf[bank_base + ASSOC_RANK + d_model];
                    if (bank_usage < min_usage) {
                        min_usage = bank_usage;
                        min_usage_bank = bank;
                    }
                    if (!found_empty && bank_usage < ASSOC_OCCUPIED_THRESHOLD) {
                        empty_bank = bank;
                        found_empty = true;
                    }
                    let cos = score / sqrt(max(key_norm * new_key_norm, 1.0e-12));
                    if (bank < 8u) { bank_scores[bank] = cos; }
                    let value_cos =
                        value_score / sqrt(max(value_norm * new_value_norm, 1.0e-12));
                    if (cos > best_cos) {
                        best_cos = cos;
                        best_value_cos = value_cos;
                        best_bank = bank;
                    }
                }

                // ── Librarian Stage 2: Competitive Slot Allocation ─────────────────────
                // All slots compete for the current token using their anchor geometry.
                // Every slot recomputes the global allocation to determine its own share.
                let slot_anchor_root_write = hist_base + slot_anchor_base(d_model, h_slots);
                var prev_sig_sq_owner = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sig_sq_owner = prev_sig_sq_owner + prev_j * prev_j;
                }
                let prev_sig_rms_owner = sqrt(prev_sig_sq_owner / max(1.0, f32(d_model)) + 1.0e-6);
                
                var max_slot_score = -1.0e30;
                for (var owner = 0u; owner < h_slots; owner = owner + 1u) {
                    let owner_off = slot_anchor_root_write + owner * d_model;
                    var anchor_sq = 0.0;
                    var dot = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let prev_j = PrevHStarBuf[prev_hstar_base + j] / max(prev_sig_rms_owner, 1.0e-6);
                        let anchor_j = AllWeights[owner_off + j];
                        anchor_sq = anchor_sq + anchor_j * anchor_j;
                        dot = dot + prev_j * anchor_j;
                    }
                    let owner_score = dot / sqrt(max(anchor_sq, 1.0e-12));
                    shared_vals[owner] = owner_score; // Temporary store in shared_vals
                    max_slot_score = max(max_slot_score, owner_score);
                }
                var slot_denom = 0.0;
                for (var owner = 0u; owner < h_slots; owner = owner + 1u) {
                    let e = exp(4.0 * (shared_vals[owner] - max_slot_score)); // Beta=4.0 for slot selection
                    shared_vals[owner] = e;
                    slot_denom = slot_denom + e;
                }
                let p_slot_i = shared_vals[slot_idx] / max(slot_denom, 1.0e-6);

                // ── Librarian Stage 3: Competitive Bank Placement ────────────────────
                // Within the slot, banks compete for the token using cosine similarity.
                let novelty = 1.0 - max(0.0, best_cos);
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                    let bank_usage = AssocBuf[bank_base + ASSOC_RANK + d_model];
                    let empty_bonus = novelty * max(0.0, 1.0 - bank_usage) * 2.0;
                    bank_scores[bank] = bank_scores[bank] + empty_bonus;
                }
                var max_bank_score = -1.0e30;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    max_bank_score = max(max_bank_score, bank_scores[bank]);
                }
                var bank_denom = 0.0;
                var bank_probs: array<f32, 8>;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let e = exp(4.0 * (bank_scores[bank] - max_bank_score)); // Beta=4.0 for bank selection
                    bank_probs[bank] = e;
                    bank_denom = bank_denom + e;
