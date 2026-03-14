# AIDEEN Bench Matrix

Perfiles fijos para comparar estabilidad/convergencia del training DEQ en GPU.

## Archivos

- `profiles.csv`: ladder de configuraciones (`512 -> 1024`)
- `run_matrix.sh`: ejecuta `stress_test` por perfil y consolida métricas

## Ejecutar

Desde la raíz del repo:

```bash
./aideen-bench/bench/run_matrix.sh
```

Opcional:

```bash
./aideen-bench/bench/run_matrix.sh ./aideen-bench/bench/profiles.csv ./aideen-bench/bench/results/manual_run
```

## Salida

Genera:

- logs por perfil: `results/<timestamp>/<profile>.log`
- tabla resumen: `results/<timestamp>/summary.csv`

Columnas clave:

- `iter20_loss`
- `hit_ratio_pct`
- `conv`
- `mode`
- `contractivity`
- `max_delta`
- `rs_cg`
- `shared`

## Criterio de promoción

Promover al siguiente `d_r` solo si:

- `conv=OK`
- `hit_ratio_pct <= 5`
- `mode=NORMAL` la mayor parte del run
- `contractivity <= 1.0` sin picos sostenidos

## Perfil v2 (tuning convergencia)

Para intentar mover `d_r=768` a `conv=OK`:

```bash
./aideen-bench/bench/run_matrix.sh ./aideen-bench/bench/profiles_v2.csv
```

