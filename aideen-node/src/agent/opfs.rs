#![cfg(target_arch = "wasm32")]

use aideen_core::agent::{AgentEvent, AgentStore};
use js_sys::Uint8Array;
use std::collections::{HashMap, VecDeque};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemWritableFileStream,
};

/// Cuántos eventos recientes mantenemos en el ring buffer en memoria.
const RECENT_CAP: usize = 256;

/// Backend WASM para Origin Private File System (OPFS).
///
/// Layout idéntico a FsAgentStore (mismo framing, misma serde):
///   <agent_id>/prefs.bin   — bincode(HashMap<String, String>)
///   <agent_id>/events.log  — [u32 LE len][bincode(AgentEvent)]*
///
/// Las escrituras son async fire-and-forget: la caché se actualiza
/// sincrónicamente y spawn_local persiste en OPFS en background.
/// Usa la API async de OPFS (createWritable) — funciona en Window y Worker.
pub struct OpfsAgentStore {
    agent_id: String,
    prefs: HashMap<String, String>,
    recent: VecDeque<AgentEvent>,
}

impl OpfsAgentStore {
    /// Abre o crea el store bajo `<agent_id>/` en OPFS.
    /// Carga prefs y los últimos eventos desde disco.
    pub async fn open(agent_id: &str) -> Result<Self, String> {
        let dir = agent_dir(agent_id).await?;

        let prefs = match read_bytes(&dir, "prefs.bin").await {
            Ok(b) => bincode::deserialize(&b).unwrap_or_default(),
            Err(_) => HashMap::new(),
        };

        let recent = match read_bytes(&dir, "events.log").await {
            Ok(b) => parse_events(&b),
            Err(_) => VecDeque::new(),
        };

        Ok(Self {
            agent_id: agent_id.to_string(),
            prefs,
            recent,
        })
    }
}

impl AgentStore for OpfsAgentStore {
    fn get_pref(&self, key: &str) -> Option<String> {
        self.prefs.get(key).cloned()
    }

    fn set_pref(&mut self, key: &str, value: String) -> Result<(), String> {
        self.prefs.insert(key.to_string(), value);
        let bytes = bincode::serialize(&self.prefs).map_err(|e| e.to_string())?;
        let id = self.agent_id.clone();
        spawn_local(async move {
            let _ = write_full(&id, "prefs.bin", bytes).await;
        });
        Ok(())
    }

    fn append_event(&mut self, event: AgentEvent) -> Result<(), String> {
        // Actualizar ring buffer en memoria
        if self.recent.len() >= RECENT_CAP {
            self.recent.pop_front();
        }
        self.recent.push_back(event.clone());

        // Serializar frame [u32 LE len][bincode(event)]
        let payload = bincode::serialize(&event).map_err(|e| e.to_string())?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&payload);

        let id = self.agent_id.clone();
        spawn_local(async move {
            let _ = append_bytes(&id, "events.log", frame).await;
        });
        Ok(())
    }

    fn recent_events(&self, limit: usize) -> Vec<AgentEvent> {
        self.recent.iter().rev().take(limit).cloned().collect()
    }
}

// ── Helpers OPFS ─────────────────────────────────────────────────────────────

async fn opfs_root() -> Result<FileSystemDirectoryHandle, String> {
    let storage = web_sys::window().ok_or("no window")?.navigator().storage();
    JsFuture::from(storage.get_directory())
        .await
        .map_err(|e| format!("OPFS getDirectory: {:?}", e))?
        .dyn_into()
        .map_err(|_| "OPFS root: not FileSystemDirectoryHandle".into())
}

async fn agent_dir(agent_id: &str) -> Result<FileSystemDirectoryHandle, String> {
    let root = opfs_root().await?;
    let opts = FileSystemGetDirectoryOptions::new();
    opts.set_create(true);
    JsFuture::from(root.get_directory_handle_with_options(agent_id, &opts))
        .await
        .map_err(|e| format!("OPFS getDirectoryHandle({}): {:?}", agent_id, e))?
        .dyn_into()
        .map_err(|_| "OPFS agent_dir: not FileSystemDirectoryHandle".into())
}

async fn read_bytes(dir: &FileSystemDirectoryHandle, name: &str) -> Result<Vec<u8>, String> {
    // Sin create:true → devuelve error si el archivo no existe (el caller lo trata como vacío)
    let fh: FileSystemFileHandle = JsFuture::from(dir.get_file_handle(name))
        .await
        .map_err(|e| format!("OPFS getFileHandle({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS read: not FileSystemFileHandle".into())?;

    let file: web_sys::File = JsFuture::from(fh.get_file())
        .await
        .map_err(|e| format!("OPFS getFile({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS read: not File".into())?;

    let ab: js_sys::ArrayBuffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|e| format!("OPFS arrayBuffer({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS read: not ArrayBuffer".into())?;

    Ok(Uint8Array::new(&ab).to_vec())
}

/// Sobreescribe `name` con `bytes` completo (para prefs.bin).
async fn write_full(agent_id: &str, name: &str, bytes: Vec<u8>) -> Result<(), String> {
    let dir = agent_dir(agent_id).await?;

    let opts = FileSystemGetFileOptions::new();
    opts.set_create(true);
    let fh: FileSystemFileHandle = JsFuture::from(dir.get_file_handle_with_options(name, &opts))
        .await
        .map_err(|e| format!("OPFS getFileHandle({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS write_full: not FileSystemFileHandle".into())?;

    // keepExistingData: false (por defecto) → trunca al abrir
    let writable: FileSystemWritableFileStream = JsFuture::from(fh.create_writable())
        .await
        .map_err(|e| format!("OPFS createWritable({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS write_full: not FileSystemWritableFileStream".into())?;

    let arr = Uint8Array::from(bytes.as_slice());
    JsFuture::from(
        writable
            .write_with_array_buffer_view(arr.unchecked_ref())
            .map_err(|e| format!("OPFS write({}): {:?}", name, e))?,
    )
    .await
    .map_err(|e| format!("OPFS write await({}): {:?}", name, e))?;

    JsFuture::from(writable.close())
        .await
        .map_err(|e| format!("OPFS close({}): {:?}", name, e))?;

    Ok(())
}

/// Agrega `bytes` al final de `name` (para events.log).
async fn append_bytes(agent_id: &str, name: &str, bytes: Vec<u8>) -> Result<(), String> {
    let dir = agent_dir(agent_id).await?;

    let opts = FileSystemGetFileOptions::new();
    opts.set_create(true);
    let fh: FileSystemFileHandle = JsFuture::from(dir.get_file_handle_with_options(name, &opts))
        .await
        .map_err(|e| format!("OPFS getFileHandle({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS append: not FileSystemFileHandle".into())?;

    // Leer tamaño actual para seek al EOF
    let file: web_sys::File = JsFuture::from(fh.get_file())
        .await
        .map_err(|e| format!("OPFS getFile({}): {:?}", name, e))?
        .dyn_into()
        .map_err(|_| "OPFS append: not File".into())?;
    let eof = file.size() as u64;

    // keepExistingData: true → no trunca; seek al EOF para append puro
    let create_opts = FileSystemCreateWritableOptions::new();
    create_opts.set_keep_existing_data(true);
    let writable: FileSystemWritableFileStream =
        JsFuture::from(fh.create_writable_with_options(&create_opts))
            .await
            .map_err(|e| format!("OPFS createWritable({}): {:?}", name, e))?
            .dyn_into()
            .map_err(|_| "OPFS append: not FileSystemWritableFileStream".into())?;

    JsFuture::from(
        writable
            .seek_with_f64(eof as f64)
            .map_err(|e| format!("OPFS seek({}): {:?}", name, e))?,
    )
    .await
    .map_err(|e| format!("OPFS seek await({}): {:?}", name, e))?;

    let arr = Uint8Array::from(bytes.as_slice());
    JsFuture::from(
        writable
            .write_with_array_buffer_view(arr.unchecked_ref())
            .map_err(|e| format!("OPFS write({}): {:?}", name, e))?,
    )
    .await
    .map_err(|e| format!("OPFS write await({}): {:?}", name, e))?;

    JsFuture::from(writable.close())
        .await
        .map_err(|e| format!("OPFS close({}): {:?}", name, e))?;

    Ok(())
}

/// Parsea events.log y devuelve los últimos RECENT_CAP eventos en orden FIFO.
fn parse_events(bytes: &[u8]) -> VecDeque<AgentEvent> {
    let mut events = VecDeque::new();
    let mut cursor = 0usize;
    while cursor + 4 <= bytes.len() {
        let len = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]) as usize;
        cursor += 4;
        if cursor + len > bytes.len() {
            break;
        }
        if let Ok(ev) = bincode::deserialize::<AgentEvent>(&bytes[cursor..cursor + len]) {
            events.push_back(ev);
        }
        cursor += len;
    }
    // Conservar solo los últimos RECENT_CAP (si el log creció más allá del cap)
    while events.len() > RECENT_CAP {
        events.pop_front();
    }
    events
}
