# Guía de Estabilidad y Damping en AIDEEN

Esta guía explica los mecanismos físicos que mantienen al motor DEQ estable bajo carga y cómo sintonizar los parámetros de ejecución.

## 1. El Solver Picard y el Damping (β)

AIDEEN utiliza iteraciones de Picard para encontrar el punto fijo del razonamiento ($H^*$). La estabilidad de este proceso se controla mediante el factor de **Damping** (o $\beta$-relaxation).

### Ecuación de Actualización:
$$H_{next} = \beta \cdot f(H_{curr}) + (1 - \beta) \cdot H_{curr}$$

*   **Damping Alto (0.6 - 0.8)**: El modelo es extremadamente estable pero converge más lento. Ideal para tareas de lógica compleja o multi-binding donde las señales son ruidosas.
*   **Damping Bajo (0.1 - 0.3)**: El modelo es muy rápido pero puede oscilar o divergir si los pesos no están perfectamente normalizados.

> [!TIP]
> Si ves NaNs o Infs en los logs durante el entrenamiento, lo primero que debes hacer es bajar el damping (ej. a `0.4`) mediante la flag `AIDEEN_DAMPING=0.4`.

## 2. Normalización Estructural

Para que AIDEEN no sea un "modelo de juguete", implementamos protecciones matemáticas en el flujo de datos:

### Logit Clipping
Los scores de la memoria asociativa se limitan a `[-25.0, 25.0]` antes de entrar al Softmax.
*   **Por qué**: Sin esto, una "llave" muy fuerte genera un valor exponencial infinito, lo que rompe el gradiente y el forward pass.

### RMSNorm on Associative Context
Toda señal recuperada de la memoria se normaliza por su propia magnitud (RMS).
*   **Por qué**: Esto garantiza que la "energía" inyectada al DEQ sea constante. El modelo no puede "gritar" más fuerte de lo que el motor de razonamiento puede procesar.

## 3. Flags de Ejecución (Modos)

| Variable | Valor Típico | Descripción |
| :--- | :--- | :--- |
| `AIDEEN_DAMPING` | `0.4` - `0.7` | Estabilidad del solver Picard. |
| `AR_PAIRS_PER_SEQ` | `1` o `2` | Cantidad de asociaciones simultáneas (Multi-binding). |
| `AR_AUDIT` | `1` | Activa telemetría detallada de memoria (Entropía, MaxProb). |
| `ASSOC_ADDR_GRAD_SCALE` | `1024` | (Interno) Calibración de la velocidad de aprendizaje de la memoria. |

## 4. Evolución de la Inteligencia vs. Estabilidad

En AIDEEN, **estabilidad = capacidad**.
Un modelo inestable no puede aprender porque su gradiente es ruido. Un modelo calibrado (con el damping y la normalización correctos) puede mantener representaciones coherentes a lo largo de miles de tokens, lo que permite el surgimiento de comportamientos complejos como el razonamiento deductivo.
