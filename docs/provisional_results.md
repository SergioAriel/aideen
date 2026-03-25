### [2026-03-20] Baseline provisional: V dinámica estable (post V-normalization fix)
**Contexto**: estabilidad DEQ con V dinámica y shader estabilizado.
**Objetivo**: eliminar DEQ-INVALID y mantener contr < 1 sin fijar V.

**Configuración exacta**
- Cmd: `<no registrado — run desde IDE>`
- Env: `<no registrado>`
- Seed: 7 / 11 / 13 / 42
- Iters: 100
- Dataset: stress_test
- Perfil: STABLE
- Código: cambios post “V-normalization fix” (archivos no listados en logs)

**Resultados**
- mode/conv: NORMAL / OK
- contr/maxΔ: contr≈0.20 (antes 1.66), iters≈7 (antes 46)
- hist/inj: ≈0.079 estable
- DEQ-INVALID: 0 en los 4 seeds

**Criterios de invalidez**
- Si reaparece DEQ-INVALID en cualquiera de esos seeds bajo misma config.
- Si cambia el shader DEQ o la normalización de V, revalidar.
- Si no se puede reproducir con un comando explícito, esta entrada se considera provisional no verificada.

**Alcance**
- Válido solo para stress_test y estas semillas; no implica default.
