# Módulo local: `engine` (Motor de Cómputo GPU)

## 📌 Responsabilidades
*   Recibir tensores del `loxi-backbone`.
*   Cargarlos en memoria VRAM utilizando `wgpu`.
*   Ejecutar los shaders `WGSL`.

## 🚫 Restricciones (Constitucionales)
*   **NO** decide lógica.
*   **NO** conoce de ética, control, memoria ni razonamiento.
*   **NO** conoce nada de la capa de red.
*   **NO** importa dependencias del módulo `system`.
