import json
import os

log_path = '/Users/sergiosolis/.gemini/antigravity/brain/7bd53936-64b2-45ff-b7f7-6a063319f745/.system_generated/logs/overview.txt'
output_dir = '/Users/sergiosolis/Programacion/aideen/scratch/recovery'
os.makedirs(output_dir, exist_ok=True)

critical_files = [
    'deq_slot_attn_unified_clean.wgsl',
    'fused_deq_update.wgsl',
    'staged_adjoint_picard_clean.wgsl'
]

with open(log_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            step = data.get('step_index', 0)
            if step >= 6772: # Only interested in what happened before the disaster
                continue
                
            tool_calls = data.get('tool_calls', [])
            for call in tool_calls:
                name = call.get('name')
                if name in ['replace_file_content', 'multi_replace_file_content', 'write_to_file']:
                    args = call.get('args', {})
                    target = args.get('TargetFile', '')
                    
                    if any(f in target for f in critical_files):
                        content = args.get('ReplacementContent', args.get('CodeContent', ''))
                        if not content: continue
                        
                        file_label = target.split('/')[-1]
                        out_name = f"step_{step}_{file_label}"
                        with open(os.path.join(output_dir, out_name), 'w') as out:
                            out.write(content)
                        print(f"Recovered {out_name}")
        except Exception as e:
            continue
