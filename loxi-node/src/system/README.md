# Módulo local: `system` (Bucle del Sistema / Orquestador)

## 📌 Responsabilidades
*   Manejar el bucle principal (`loop`) del nodo de inferencia.
*   Definir el orden estricto de ejecución, basándose en los contratos puros de `loxi-core`:
    1.  `Reasoning::step`
    2.  `Control::decide`
    3.  `Ethics::violates`
    4.  `Integración matemática` (S ← S + tanh(α · Δ))

## 🚫 Restricciones (Constitucionales)
*   **NO** conoce detalles de implementación locales de la GPU.
*   **NO** conoce los detalles del puerto, sockets ni del protocolo de red (delega estas tareas en `engine` y `network` respectivamente).
