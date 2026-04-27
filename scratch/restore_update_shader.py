
import re

# Read the original file from git
with open('/tmp/fused_deq_update_original.wgsl', 'r') as f:
    content = f.read()

# The original has the OLD layout (hist_delta_base full-rank).
# We need to update ONLY the two functions that changed for the new per-slot layout:
#   hist_delta_bias_base: OLD = hist_delta_base(d, h_slots) + h_slots * d * d
#                         NEW = w_v_write_base(d, h_slots) + h_slots * RETAIN_RANK * d
#   hist_selective_flag_base: OLD = hist_delta_bias_base(...) + d
#                             NEW = hist_delta_bias_base(...) + h_slots * d
# We also need to add w_k_write_base and w_v_write_base before hist_delta_bias_base.

# 1. Remove the old hist_delta_base and hist_delta_bias_base definitions
content = re.sub(
    r'fn hist_delta_base\(.*?\) -\> u32 \{[^\n]+\}\n\n',
    '', content
)
content = re.sub(
    r'fn hist_delta_bias_base\(.*?\) -\> u32 \{[^\n]+\}\n',
    '', content
)
# 2. Fix hist_selective_flag_base: old was +d, new must be +h_slots*d
content = content.replace(
    'fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 {\n    return hist_delta_bias_base(d, h_slots) + d;\n}',
    'fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 {\n    return hist_delta_bias_base(d, h_slots) + h_slots * d;\n}'
)
# 3. Insert w_k_write_base, w_v_write_base, and new hist_delta_bias_base after slot_anchor_base
new_write_bases = """fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * RETAIN_RANK; }
fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 { return w_v_write_base(d, h_slots) + h_slots * RETAIN_RANK * d; }
"""
# Insert after the slot_anchor_base definition
content = content.replace(
    'fn slot_anchor_base(d: u32, h_slots: u32) -> u32 {\n    return hist_gate_base(d, h_slots) + h_slots;\n}\n\n',
    'fn slot_anchor_base(d: u32, h_slots: u32) -> u32 {\n    return hist_gate_base(d, h_slots) + h_slots;\n}\n\n' + new_write_bases + '\n'
)
# 4. Add RETAIN_RANK and ASSOC_RANK constants if not present
if 'const RETAIN_RANK: u32' not in content:
    content = content.replace(
        '// AllWeights layout offset functions',
        'const RETAIN_RANK: u32 = 32u;\nconst ASSOC_RANK: u32 = 32u;\n\n// AllWeights layout offset functions'
    )

with open('aideen-backbone/src/shaders/fused_deq_update.wgsl', 'w') as f:
    f.write(content)

print(f"Done. Lines: {content.count(chr(10))}")
