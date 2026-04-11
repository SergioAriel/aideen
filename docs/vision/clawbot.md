# Clawbot — Bot con Memoria Persistente por Usuario

## Visión

Clawbot es una aplicación de AIDEEN como asistente conversacional donde cada usuario tiene su propia memoria persistente. El modelo base es compartido e inmutable en runtime. La adaptación por usuario ocurre exclusivamente a través de la memoria plástica `M_user`.

## Stack de capas

```
W_base  (AIDEEN preentrenado)
└── Sabe lenguaje, razonamiento, coherencia
└── Igual para todos los usuarios y dispositivos
└── Solo cambia con nuevas versiones del modelo

W_bot   (fine-tuning sobre W_base)
└── Sabe quién es clawbot
└── Sabe qué puede hacer y cómo responde
└── Conoce su rol, herramientas, límites
└── Igual para todos los usuarios
└── Solo cambia con nuevas versiones del bot

M_user  (memoria por usuario, persiste entre sesiones)
└── Aprende cómo habla ese usuario
└── Aprende sus preferencias y proyectos
└── Aprende su contexto y estilo
└── Completamente independiente entre usuarios
└── Nunca se sincroniza globalmente
```

## Lifecycle de M_user

```
Nueva sesión:
  si existe M_user_X.bin → cargar
  si no → inicializar en ceros

Durante la sesión:
  M_user evoluciona token a token (igual que hoy con MState)
  el usuario experimenta un bot que ya lo conoce

Al cerrar la sesión:
  guardar M_user_X.bin

Actualización del modelo base:
  W_base y W_bot se actualizan
  M_user de cada usuario no se toca
  el bot evoluciona pero recuerda lo aprendido
```

## Distribución

Los pesos W_base y W_bot viven en la nube y se sincronizan a los dispositivos. AIDEEN es distribuido — cada dispositivo corre su propia instancia del modelo.

Clawbot es independiente del dispositivo. La sesión puede correr en cualquier dispositivo que tenga los pesos sincronizados. M_user se carga y guarda por usuario, no por dispositivo.

```
Nube
├── W_base.aidn      ← sincronizado a todos los dispositivos
└── W_bot.aidn       ← sincronizado a todos los dispositivos

Dispositivo A / B / C
├── W_base (sync)
├── W_bot  (sync)
└── M_users/
    ├── user_123.bin
    ├── user_456.bin
    └── ...
```

## Pre-training del bot

Clawbot requiere un fine-tuning mínimo antes de deployment:
- **Identidad**: quién es, cómo se llama, qué puede hacer
- **Instrucción**: cómo seguir instrucciones, formatear respuestas, manejar límites
- **Herramientas**: qué acciones puede tomar y cómo reportarlas

Este fine-tuning es sobre W_base y produce W_bot. No toca M — M empieza en ceros para cada usuario nuevo y crece sola con el uso.

## Qué aprende M_user con el tiempo

Sin intervención explícita, solo por interacción normal:
- Vocabulario y estilo de escritura del usuario
- Proyectos y contextos recurrentes
- Preferencias de formato y detalle
- Patrones de razonamiento del usuario

Esto emerge del mecanismo de retain gate — el modelo aprende qué retener de las interacciones pasadas porque esa retención mejora las predicciones futuras.

## Privacidad

M_user contiene información aprendida de las interacciones del usuario. Consideraciones:
- M_user nunca se comparte entre usuarios
- El usuario puede solicitar reset de su M (ceros)
- Las actualizaciones de W_base no leen M_user
- M_user no contiene texto plano — es una representación latente

## Estado actual vs. lo que falta

| Componente | Estado |
|---|---|
| MState por chunk | Implementado |
| Persistencia de MState entre sesiones | Pendiente (solo guardar/cargar el buffer) |
| Fine-tuning de W_bot | Pendiente (requiere corpus de instrucción) |
| M_user por usuario en disco | Pendiente |
| Neuroplasticidad real (W_eff plástico) | Roadmap futuro |
