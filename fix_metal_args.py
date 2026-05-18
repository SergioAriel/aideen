import re

file_path = "aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl"
with open(file_path, "r") as f:
    content = f.read()

variables = [
    "max_delta_seen", "max_m_delta_seen", "max_a_delta_seen", 
    "sum_self_assign_seen", "sum_assign_entropy_seen", "sum_slot_move_seen",
    "max_err_h_seen", "max_err_m_seen", "max_z_seen", "max_update_ratio_seen",
    "max_memctx_rms_seen", "max_memctx_to_signal_seen", "rescue_count_seen",
    "rescue_recovered_seen", "dead_slot_seen", "write_saturation_seen",
    "last_delta", "max_contractivity", "max_h_seen", "hhist_gamma_wg"
]

# 1. Create the struct definition
struct_def = "struct DiagScalars {\n"
for v in variables:
    struct_def += f"    {v}: f32,\n"
struct_def += "}\nvar<workgroup> wg_diags: DiagScalars;\n"

# 2. Remove the old declarations
for v in variables:
    content = re.sub(fr"var<workgroup>\s+{v}\s*:\s*f32\s*;\n", "", content)

# 3. Insert the struct where the first variable used to be
content = content.replace("var<workgroup> slot_coord_prev: array<f32, MAX_SLOTS>;\n", 
                          "var<workgroup> slot_coord_prev: array<f32, MAX_SLOTS>;\n" + struct_def)

# 4. Replace usages (using word boundaries to avoid partial matches)
for v in variables:
    content = re.sub(fr"\b{v}\b", f"wg_diags.{v}", content)

with open(file_path, "w") as f:
    f.write(content)

print("Refactoring complete.")
