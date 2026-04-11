# Propuesta de Proyecto de I+D — CDTI

## Convocatoria

Proyectos de I+D — Centro para el Desarrollo Tecnologico y la Innovacion (CDTI)

---

## 1. Resumen Ejecutivo

> [PLACEHOLDER: Un parrafo que resuma el proyecto. Debe cubrir: el problema (dependencia europea de proveedores de IA estadounidenses), la solucion propuesta (AIDEEN, motor de IA descentralizado basado en DEQ+Fixed-Point Memory, escrito en Rust), la diferenciacion tecnica (eficiencia parametrica O(1), inferencia en navegador via WebGPU, red P2P con gobernanza criptografica), y el resultado esperado (infraestructura de IA soberana, de codigo abierto, ejecutable en hardware de consumo).]

---

## 2. Planteamiento del Problema

### 2.1 Dependencia tecnologica de la UE

La Union Europea depende estructuralmente de proveedores estadounidenses para capacidades avanzadas de inteligencia artificial. Los modelos de lenguaje mas utilizados (GPT-4, Claude, Gemini) son desarrollados, alojados y controlados por empresas con sede en Estados Unidos. Las organizaciones europeas que integran estas capacidades en sus productos y servicios operan bajo tres riesgos simultaneos:

1. **Riesgo de soberania**: Los datos de inferencia transitan por infraestructura extranjera, sujeta a jurisdicciones no europeas.
2. **Riesgo de continuidad**: El acceso depende de decisiones comerciales unilaterales (cambios de precios, restricciones de uso, discontinuacion de modelos).
3. **Riesgo de concentracion**: El mercado converge hacia un oligopolio de 3-4 proveedores, eliminando la competencia real.

### 2.2 Barrera de acceso por hardware

Los modelos de IA actuales requieren GPUs de gama alta para inferencia (un modelo de 70B parametros necesita mas de 35 GB de VRAM). Esto excluye a:

- Pymes sin presupuesto para infraestructura cloud.
- Administraciones publicas con restricciones de datos.
- Paises en desarrollo sin centros de datos avanzados.
- Usuarios individuales sin hardware especializado.

### 2.3 Oportunidad

Existe una ventana de oportunidad para desarrollar una arquitectura de IA fundamentalmente mas eficiente que permita inferencia en hardware de consumo, sin depender de infraestructura cloud centralizada.

---

## 3. Solucion Propuesta

### 3.1 AIDEEN: Motor de IA descentralizado

AIDEEN es un motor de inferencia y entrenamiento de IA desarrollado integramente en Rust. Reemplaza la arquitectura transformer dominante por una combinacion de:

- **Deep Equilibrium Models (DEQ)**: Un unico bloque computacional que itera hasta convergencia (punto fijo), en lugar de 24-96 capas apiladas. Complejidad parametrica O(1) frente a O(N).
- **Fixed-Point Memory State Space Models (SSM)**: Memoria temporal selectiva integrada en el bloque DEQ, con atencion cruzada entre slots de razonamiento.
- **Iteracion de Picard con normalizacion espectral**: Garantia matematica de convergencia del punto fijo.

### 3.2 Arquitectura de red

- **P2P sobre QUIC/WebTransport**: Los nodos se comunican directamente, sin servidor central obligatorio.
- **Gobernanza criptografica zero-trust**: Delegacion de claves con firma Ed25519, anti-replay por epocas, actualizaciones encadenadas.
- **Protocolo v1 congelado**: Constantes de protocolo inmutables dentro de una version, garantizando interoperabilidad.

### 3.3 Compatibilidad de hardware

- **GPU agnostico**: Compilacion a Metal (Apple), Vulkan (Linux/Android), DX12 (Windows), WebGPU (navegador) via `wgpu`.
- **Inferencia en navegador**: WebGPU + WebAssembly permiten inferencia directa en cualquier navegador moderno, sin instalacion, sin envio de datos a servidores externos.

### 3.4 Seguridad etica

- **EthicsKernel**: Modulo de seguridad no entrenable, aplicado a todas las salidas. No recibe gradientes, no puede ser modificado por entrenamiento ni configuracion.

---

## 4. Estado del Arte y Diferenciacion

### 4.1 Modelos open-source existentes

| Proyecto | Arquitectura | Parametros | Hardware minimo | Descentralizado |
|----------|-------------|------------|-----------------|-----------------|
| Llama 3 (Meta) | Transformer | 8B-405B | 16-810 GB VRAM | No |
| Mistral (Mistral AI) | Transformer MoE | 7B-8x22B | 14-176 GB VRAM | No |
| Phi-3 (Microsoft) | Transformer | 3.8B-14B | 8-28 GB VRAM | No |
| **AIDEEN** | **DEQ + Fixed-Point Memory SSM** | **O(1) reusable** | **iGPU / navegador** | **Si** |

### 4.2 Diferenciacion tecnica

1. **Parametros**: Los modelos transformer escalan linealmente con la profundidad. AIDEEN reutiliza un unico bloque, reduciendo drasticamente el numero de parametros necesarios para una capacidad equivalente.
2. **Memoria de entrenamiento**: Los transformers almacenan activaciones de todas las capas para backpropagation. AIDEEN usa diferenciacion implicita (teorema de la funcion implicita), reduciendo la memoria de entrenamiento de O(N) a O(1).
3. **Inferencia en navegador**: Ninguno de los modelos open-source existentes ofrece inferencia nativa en navegador via WebGPU sin servidor backend.
4. **Descentralizacion nativa**: AIDEEN incluye un protocolo P2P con gobernanza criptografica de serie, no como complemento posterior.
5. **Seguridad no negociable**: El EthicsKernel es un invariante arquitectonico, no un filtro de post-procesamiento que pueda desactivarse.

### 4.3 Publicaciones relevantes

- Bai et al., "Deep Equilibrium Models" (NeurIPS 2019)
- Gu & Dao, "Fixed-Point Memory: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Bai et al., "Stabilizing Equilibrium Models by Jacobian Regularization" (ICML 2021)

---

## 5. Plan de Trabajo

### WP1: Pipeline de entrenamiento y validacion del modelo (meses 1-4)

| Tarea | Descripcion | Entregable |
|-------|-------------|------------|
| T1.1 | Finalizacion del pipeline de entrenamiento de 4 fases (Decomposer, Backbone, Expertos Federados, Destilacion) | Pipeline funcional end-to-end |
| T1.2 | Entrenamiento del backbone DEQ+Fixed-Point Memory con corpus multilingual (BPE 64K) | Pesos del backbone validados |
| T1.3 | Entrenamiento de al menos 6 dominios expertos (math, code, logic, NLP, science, creative) | Pesos de expertos + metricas de calidad |
| T1.4 | Validacion contra benchmarks estandar (perplexity, downstream tasks) | Informe comparativo vs Llama/Mistral iso-parametro |

**Hito M1 (mes 4)**: Modelo entrenado con perplexity competitiva en benchmarks estandar.

### WP2: Demo en navegador y red P2P (meses 3-6)

| Tarea | Descripcion | Entregable |
|-------|-------------|------------|
| T2.1 | Compilacion WASM del motor de inferencia con WebGPU | Binary WASM funcional |
| T2.2 | Interfaz web de demo (chat basico en navegador) | Demo publica accesible por URL |
| T2.3 | Red P2P funcional: descubrimiento de nodos, delegacion de claves, actualizaciones firmadas | Red de al menos 3 nodos interoperando |
| T2.4 | Integracion WebTransport para nodos en navegador | Nodo ligero en navegador conectado a la red |

**Hito M2 (mes 6)**: Demo publica de inferencia en navegador + red P2P funcional.

### WP3: Benchmarking, publicacion y comunidad (meses 5-8)

| Tarea | Descripcion | Entregable |
|-------|-------------|------------|
| T3.1 | Benchmark iso-parametro DEQ+SSM vs Transformer (calidad, latencia, memoria) | Datos reproducibles + scripts |
| T3.2 | Redaccion de articulo tecnico para conferencia (NeurIPS/ICML/EMNLP) | Preprint en arXiv |
| T3.3 | Publicacion del repositorio como open-source | Repositorio publico con documentacion |
| T3.4 | Programa de contribuidores y documentacion tecnica para la comunidad | Guia de contribucion + primeros PRs externos |

**Hito M3 (mes 8)**: Articulo enviado + repositorio publico + comunidad inicial.

---

## 6. Equipo

| Rol | Perfil | Dedicacion |
|-----|--------|------------|
| Investigador principal / Desarrollador 1 | Ingeniero de software con experiencia en Rust, GPU computing (wgpu/WGSL), deep learning, y sistemas distribuidos. Responsable de la arquitectura DEQ, motor GPU, y protocolo P2P. | 100% |
| Desarrollador 2 | Ingeniero de software con experiencia en Rust, entrenamiento de modelos de lenguaje, y desarrollo web (WASM/WebGPU). Responsable del pipeline de entrenamiento, benchmarks, y demo en navegador. | 100% |

Ambos desarrolladores tienen dominio completo de Rust, lo que permite contribuciones cruzadas en cualquier componente del sistema.

---

## 7. Presupuesto Estimado

| Partida | Descripcion | Coste (EUR) |
|---------|-------------|-------------|
| Personal | 2 desarrolladores x 8 meses x [salario mensual] | [COMPLETAR] |
| Computacion | GPU cloud para entrenamiento (fases 1-2), hardware de test | [COMPLETAR] |
| Viajes | Conferencias (1-2 conferencias internacionales), reuniones con evaluadores | [COMPLETAR] |
| Otros costes directos | Licencias, servicios cloud auxiliares, dominio/hosting demo | [COMPLETAR] |
| Costes indirectos | Overhead (15-25% segun normativa CDTI) | [COMPLETAR] |
| **Total** | | **[COMPLETAR]** |

*Nota: El presupuesto de computacion se optimiza gracias a la eficiencia parametrica del DEQ. El entrenamiento de AIDEEN requiere una fraccion del coste GPU de un transformer equivalente.*

---

## 8. Impacto Esperado

### 8.1 Soberania tecnologica de la UE

- Infraestructura de IA desarrollada en Europa, bajo jurisdiccion europea.
- Codigo fuente abierto que permite auditoria completa por reguladores y ciudadanos.
- Sin dependencia de proveedores cloud estadounidenses para inferencia.

### 8.2 Accesibilidad

- Inferencia en hardware de consumo (GPUs integradas, portatiles, smartphones).
- Inferencia en navegador sin instalacion: acceso universal con una URL.
- Reduccion dramatica del coste de acceso a IA avanzada para pymes y administraciones publicas.

### 8.3 Contribucion al ecosistema open-source

- Publicacion de una arquitectura novedosa (DEQ+Fixed-Point Memory) completamente en Rust.
- Protocolo abierto y versionado para redes de IA descentralizadas.
- Benchmark reproducible que compara DEQ vs Transformer en condiciones controladas.
- Articulo cientifico revisado por pares.

### 8.4 Alineacion con prioridades europeas

- **EU AI Act**: Arquitectura auditable con modulo de seguridad no desactivable (EthicsKernel).
- **Soberania digital**: Alternativa europea a infraestructura de IA controlada por terceros paises.
- **Agenda Digital Espanola**: Contribucion a la capacidad tecnologica nacional en IA.

---

## 9. Indicadores de Exito (KPIs)

| Indicador | Metrica | Objetivo |
|-----------|---------|----------|
| Calidad del modelo | Perplexity en benchmarks estandar | Competitivo con Llama/Mistral de parametros similares |
| Eficiencia parametrica | Ratio calidad/parametros vs transformer iso-parametro | Mejora >= 2x en calidad/parametro |
| Latencia de inferencia | Tokens/segundo en iGPU (Radeon 780M o Apple M1) | >= 10 tokens/s |
| Inferencia en navegador | Demo funcional en Chrome/Firefox/Safari | Si, publicamente accesible |
| Red P2P | Nodos interoperando con protocolo v1 | >= 3 nodos |
| Publicacion cientifica | Articulo aceptado o en revision | >= 1 preprint en arXiv |
| Open-source | Repositorio publico con contribuciones externas | >= 5 contribuidores externos |
| Seguridad | EthicsKernel activo y no desactivable en todas las configuraciones | 100% cobertura de salidas |

---

## Anexos

- Anexo A: Resumen tecnico de AIDEEN (ver `docs/grants/technical-summary.md`)
- Anexo B: Especificacion del protocolo v1 (ver `docs/protocol_v1.md`)
- Anexo C: Decisiones de arquitectura (ver `ARCHITECTURE_DECISIONS.md`)
