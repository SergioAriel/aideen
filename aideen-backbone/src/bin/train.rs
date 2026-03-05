//! Binary de entrenamiento de AIDEEN.
//!
//! Uso:
//!   cargo run --release --features wgpu -p aideen-backbone --bin train
//!
//! Entrena el modelo completo (Embeddings + DEQ + LmHead) con GPU (Metal)
//! y un corpus de ~5K tokens.

use aideen_backbone::{tokenizer::Tokenizer, trainer::Trainer};
use aideen_core::state::ArchitectureConfig;

const CORPUS: &str = "\
la inteligencia artificial distribuida razona en equilibrio profundo. \
cada neurona artificial converge a un punto fijo estable y robusto. \
aideen es una red de neuronas artificiales distribuidas en el borde. \
el razonamiento emerge de la convergencia del equilibrio profundo. \
cada nodo procesa informacion y comparte conocimiento con sus pares. \
la red neuronal distribuida aprende sin servidor central alguno. \
inteligencia artificial en cada dispositivo del mundo conectado. \
aideen distribuye razonamiento a traves de atractores neuronales. \
convergencia profunda genera representaciones ricas de significado. \
el equilibrio emerge cuando cada neurona encuentra su punto fijo. \
la red aprende de forma distribuida sin necesitar un coordinador. \
cada iteracion del equilibrio profundo refina el razonamiento. \
aideen es inteligencia artificial distribuida en equilibrio. \
las redes neuronales profundas aprenden patrones complejos del mundo. \
el aprendizaje automatico transforma datos en conocimiento util. \
cada capa de la red extrae caracteristicas cada vez mas abstractas. \
la propagacion hacia atras ajusta los pesos para reducir el error. \
un modelo bien entrenado generaliza a datos que nunca ha visto. \
las funciones de activacion introducen no linealidad en la red. \
el descenso por gradiente busca el minimo de la funcion de perdida. \
la regularizacion previene el sobreajuste a los datos de entrenamiento. \
los transformadores revolucionaron el procesamiento del lenguaje natural. \
la atencion permite al modelo enfocarse en las partes relevantes. \
la tokenizacion convierte texto en secuencias de numeros enteros. \
los embeddings representan palabras como vectores en un espacio continuo. \
el entrenamiento distribuido acelera el proceso con multiples maquinas. \
la normalizacion estabiliza el entrenamiento de redes profundas. \
cada neurona computa una combinacion lineal seguida de una activacion. \
los optimizadores adaptativos ajustan la tasa de aprendizaje por parametro. \
la perdida de entropia cruzada mide la diferencia entre distribuciones. \
un punto fijo es un estado que no cambia bajo la transformacion. \
el equilibrio profundo encuentra puntos fijos de redes infinitamente profundas. \
la diferenciacion implicita calcula gradientes sin almacenar activaciones. \
el gradiente conjugado resuelve sistemas lineales de forma eficiente. \
la normalizacion espectral controla la norma de los pesos de la red. \
la amortiguacion de picard asegura convergencia del punto fijo. \
aideen usa mamba como bloque fundamental de razonamiento secuencial. \
mamba es un modelo de espacio de estados selectivo y eficiente. \
la atencion entre slots permite comunicacion entre fichas semanticas. \
cada slot almacena un aspecto diferente del razonamiento actual. \
la convergencia se detecta cuando la norma del cambio es pequena. \
las redes de expertos distribuyen conocimiento especializado. \
cada experto se enfoca en un dominio particular del conocimiento. \
el enrutador selecciona los expertos mas relevantes para cada consulta. \
la agregacion combina las respuestas de multiples expertos. \
la deteccion de valores atipicos protege contra expertos maliciosos. \
la criptografia asegura la integridad de las actualizaciones de pesos. \
las firmas digitales verifican la autenticidad del remitente. \
los nonces previenen ataques de repeticion en el protocolo. \
la gobernanza controla quien puede modificar la red distribuida. \
el protocolo quic proporciona comunicacion segura entre nodos. \
la confianza se establece en el primer contacto entre pares. \
cada nodo mantiene un registro de pares conocidos y confiables. \
las actualizaciones de modelo se distribuyen de forma segura. \
el critico evalua la calidad de los pesos antes de distribuirlos. \
un checkpoint permite guardar y restaurar el estado del modelo. \
la red peer to peer elimina la necesidad de un servidor central. \
cada dispositivo contribuye su capacidad de computo a la red. \
la inferencia en el borde reduce la latencia y mejora la privacidad. \
";

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════╗");
    println!("║     AIDEEN — Training Engine v3              ║");
    println!("║     D_R=512 · Embeddings · Cosine LR         ║");
    println!("╚═══════════════════════════════════════════════╝");
    println!();

    // GPU detection
    #[cfg(feature = "wgpu")]
    {
        use aideen_backbone::gpu_backend::WgpuBlockBackend;
        match WgpuBlockBackend::new_blocking() {
            Some(_) => println!("  GPU: Metal detected ✅"),
            None => println!("  GPU: not available, using CPU"),
        }
    }
    #[cfg(not(feature = "wgpu"))]
    println!("  GPU: disabled (compile with --features wgpu)");

    // Tokenizer
    let config = ArchitectureConfig::default();
    let tok = Tokenizer::from_text(CORPUS, config);
    let tokens = tok.encode(CORPUS);
    println!(
        "  Corpus: {} chars → {} tokens, vocab={}",
        CORPUS.len(),
        tokens.len(),
        tok.vocab_size()
    );

    // Trainer
    let lr = 0.003;
    let epochs = 30;
    let mut trainer = Trainer::from_tokenizer(tok, lr);
    trainer.training_config.lr_min = 0.0003;
    trainer.training_config.warmup_epochs = 3;
    trainer.training_config.epochs = epochs;
    trainer.config.train_deq = true;
    trainer.config.deq_grad_scale = 0.05; // increased to allow backbone learning
    trainer.config.max_deq_iters = 8;
    trainer.config.cg_iters = 5;
    trainer.config.ctx_len = 12;

    // Activate GPU for reasoning.step() calls (Mamba SSM)
    #[cfg(feature = "wgpu")]
    {
        use aideen_backbone::gpu_backend::WgpuBlockBackend;
        if let Some(gpu) = WgpuBlockBackend::new_blocking() {
            trainer.reasoning.set_backend(gpu);
            println!("  Backend: GPU (Metal) ✅");
        } else {
            println!("  Backend: CPU (fallback)");
        }
    }

    // Train
    let t0 = std::time::Instant::now();
    trainer.train_on_tokens(&tokens, epochs, 3);
    let elapsed = t0.elapsed();

    println!();
    println!("── Resultado ───────────────────────────────────");
    println!("  Tiempo total: {:.1}s", elapsed.as_secs_f32());
    println!("  Spectral norms: {:?}", trainer.reasoning.spectral_norms());

    // Checkpoint
    let path = "model.aidn";
    match trainer.save_deq(path) {
        Ok(_) => {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            println!("  Checkpoint: {path} ({:.2} MB)", size as f64 / 1_048_576.0);
        }
        Err(e) => println!("  Error: {e}"),
    }

    // Generación
    println!();
    println!("── Generación ──────────────────────────────────");
    let prompts = [
        "la inteligencia artificial",
        "cada neurona",
        "aideen es una red",
        "el equilibrio profundo",
        "la red neuronal distribuida",
    ];
    for prompt in &prompts {
        let generated = trainer.generate(prompt, 40, 0.8, 0.9, 40, 1.1);
        println!("  \"{prompt}\" →");
        println!("    \"{generated}\"");
    }

    println!();
    println!("✅ AIDEEN training completo.");
}
