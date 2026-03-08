from datasets import load_dataset
import os
import re

# Plan de preparación de datos "Genesis" para AIDEEN
# Objetivo: Dataset híbrido (SmolTalk + Conocimiento de Rust)

def prepare_aideen_data():
    SYSTEM_PROMPT = "SYSTEM: Eres AIDEEN, una IA de arquitectura Deep Equilibrium (DEQ) con Slots de razonamiento. Eres precisa, lógica y experta en Rust."
    
    file_path = "train_aideen.txt"
    rust_dataset_path = "aideen-backbone/dataset.txt"
    
    print(f"📝 Generando {file_path}...")
    
    count_smol = 0
    count_rust = 0

    with open(file_path, "w", encoding="utf-8") as f:
        # 1. Procesar SmolTalk (Identidad y Chat)
        print("🚀 Descargando SmolTalk (Everyday Conversations)...")
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
            for example in ds:
                f.write(f"{SYSTEM_PROMPT}\n")
                for msg in example["messages"]:
                    role = "USER" if msg["role"] == "user" else "ASSISTANT"
                    content = msg["content"].replace("\n", " ").strip()
                    f.write(f"{role}: {content}\n")
                f.write("<|endoftext|>\n")
                count_smol += 1
                if count_smol >= 5000: break # Suficiente para identidad
        except Exception as e:
            print(f"⚠️ Error cargando SmolTalk, procediendo solo con Rust: {e}")

        # 2. Procesar Rust Documentación (Conocimiento Técnico)
        if os.path.exists(rust_dataset_path):
            print(f"📚 Procesando {rust_dataset_path}...")
            with open(rust_dataset_path, "r", encoding="utf-8") as rf:
                content = rf.read()
                # Dividimos por headers para crear "lecciones"
                chunks = re.split(r'## |### ', content)
                for chunk in chunks:
                    if not chunk.strip(): continue
                    lines = chunk.strip().split("\n")
                    header = lines[0]
                    body = " ".join(lines[1:]).replace("\n", " ").strip()
                    
                    if len(body) < 50: continue

                    f.write(f"{SYSTEM_PROMPT}\n")
                    f.write(f"USER: Háblame sobre {header} en Rust.\n")
                    f.write(f"ASSISTANT: {body}\n")
                    f.write("<|endoftext|>\n")
                    count_rust += 1
        else:
            print(f"❌ No se encontró {rust_dataset_path}")

    print(f"✨ ¡Listo! Dataset Genesis generado:")
    print(f"   - SmolTalk: {count_smol} ejemplos")
    print(f"   - Rust Docs: {count_rust} ejemplos")
    print(f"👉 Próximo paso: cargo run --release --features wgpu -p aideen-backbone --bin train -- --file train_aideen.txt --epochs 1")

if __name__ == "__main__":
    prepare_aideen_data()

if __name__ == "__main__":
    prepare_aideen_data()
