// Loxi V8 — Web Worker
// Owns the LoxiEngine (WASM + WebGPU). All blocking GPU work runs here.
// Main thread communicates via postMessage({ type, ...payload }).

import init, { LoxiEngine } from './pkg/loxi_compute.js';

let engine = null;

// ── Message protocol ──────────────────────────────────────────────────
// Main → Worker:
//   { type: 'init' }
//   { type: 'claim_gpu' }
//   { type: 'load_weights', orchestratorUrl: string, modelId: string, shardBytes: number }
//   { type: 'inference', tokenIds: Uint32Array }
//
// Worker → Main:
//   { type: 'log', message: string }
//   { type: 'ready' }
//   { type: 'gpu_claimed', deviceName: string }
//   { type: 'weights_loaded', tensorCount: number, weightKeys: string[] }
//   { type: 'inference_result', logits: Float32Array }
//   { type: 'error', message: string }

function post(type, payload = {}) {
    self.postMessage({ type, ...payload });
}

function log(msg) {
    post('log', { message: msg });
}

// ── Main message handler ──────────────────────────────────────────────
self.onmessage = async (event) => {
    const { type, ...data } = event.data;

    try {
        switch (type) {

            case 'init': {
                await init();
                engine = new LoxiEngine();
                post('ready');
                log('[Worker] WASM loaded and LoxiEngine allocated.');
                break;
            }

            case 'claim_gpu': {
                const result = await engine.initialize();
                log(`[Worker] ${result}`);
                post('gpu_claimed', { deviceName: engine.device_name() });
                break;
            }

            case 'load_weights': {
                // Receive pre-fetched weight bytes from main thread
                const { weightBytes } = data;
                log(`[Worker] Uploading ${(weightBytes.length / 1_048_576).toFixed(2)} MB to GPU VRAM...`);
                const result = engine.load_safetensors(weightBytes);
                const keys = JSON.parse(engine.list_weights());
                log(`[Worker] ${result}`);
                post('weights_loaded', {
                    tensorCount: engine.tensor_count(),
                    weightKeys: keys,
                });
                break;
            }

            case 'inference': {
                const { tokenIds } = data;
                log(`[Worker] Forward pass on ${tokenIds.length} token(s)...`);
                // run_inference returns a Promise<Float32Array>
                const logits = await engine.run_inference(tokenIds);
                post('inference_result', { logits });
                break;
            }

            default:
                post('error', { message: `Unknown message type: ${type}` });
        }
    } catch (err) {
        post('error', { message: String(err) });
    }
};
