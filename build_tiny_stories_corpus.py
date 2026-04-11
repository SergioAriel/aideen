#!/usr/bin/env python3
"""
build_tiny_stories_corpus.py — Descarga TinyStories y construye corpus para AIDEEN.
"""
import urllib.request
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
HF_TRAIN = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
HF_VALID = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# Usamos concatenación para evitar tokens reservados en la propia IA que genera esto
DOC_SEP = "<|" + "endoftext" + "|>"

def download_and_process(url, output_path, max_stories):
    print(f"Descargando de {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIDEEN-Builder/1.0"})
        with urllib.request.urlopen(req) as response:
            with open(output_path, "w", encoding="utf-8") as out_f:
                stories_written = 0
                current_story = []
                
                # Leemos línea por línea para no agotar la RAM
                for line in response:
                    line_str = line.decode('utf-8', errors='replace').strip()
                    
                    if line_str == DOC_SEP:
                        if current_story:
                            out_f.write(" ".join(current_story) + "\n\n" + DOC_SEP + "\n")
                            stories_written += 1
                            current_story = []
                            
                            if stories_written % 5000 == 0:
                                print(f"  {stories_written} historias procesadas...")
                                
                        if stories_written >= max_stories:
                            break
                    else:
                        if line_str:
                            current_story.append(line_str)
                            
                # Última historia si quedó pendiente
                if current_story and stories_written < max_stories:
                    out_f.write(" ".join(current_story) + "\n\n" + DOC_SEP + "\n")
                    stories_written += 1

                print(f"Terminado. {stories_written} historias guardadas en {output_path}")

    except Exception as e:
        print(f"Error descargando {url}: {e}")

if __name__ == "__main__":
    train_path = PROJECT_ROOT / "corpus_tinystories_train.txt"
    val_path = PROJECT_ROOT / "corpus_tinystories_val.txt"
    smoke_path = PROJECT_ROOT / "corpus_tinystories_smoke.txt"
    
    # 50,000 historias (aprox 20-25MB) es ideal para un entrenamiento rápido pero realista
    download_and_process(HF_TRAIN, train_path, 50000)
    
    # 5,000 historias para validación
    download_and_process(HF_VALID, val_path, 5000)
    
    # 200 historias para un "smoke test" ultra rápido
    download_and_process(HF_TRAIN, smoke_path, 200)
    
    print("\nArchivos creados:")
    for p in [train_path, val_path, smoke_path]:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f" - {p.name}: {size_mb:.2f} MB")
