# Plan de ataque Aureus v2.0 (robustez + diferenciación Chile)

## Objetivo general
Transformar el MVP en una plataforma de decisión de inversión más robusta, explicable e innovadora para el contexto chileno, manteniendo ejecución incremental y medible.

---

## Fase 1 — Fundamentos de datos y contexto real (prioridad máxima)
**Duración estimada:** 2 semanas

### Objetivos
- Reemplazar contexto hardcodeado por contexto real (CMF, BCCh, DF + News API opcional).
- Mejorar señal predictiva con más features técnicas y macro.
- Eliminar mismatch entre entrenamiento y uso en vivo en lo relacionado a contexto.

### Entregables
1. `context_service.py` con:
   - integración de `NewsEngine` real,
   - fallback robusto,
   - scoring por LLM de headlines en español,
   - caché controlada por TTL.
2. `features.py` con feature set expandido:
   - momentum multi-horizonte,
   - MACD, Bollinger, OBV, volumen relativo,
   - ATR ratio,
   - correlaciones rolling con cobre,
   - indicadores macro adicionales (Litio proxy, EEM, VIX).
3. `models.py` y `train_sentinel.py` alineados al nuevo set de features.
4. `requirements.txt` actualizado para nuevas dependencias de Fase 1.

### Criterio de éxito
- Pipeline de generación de features ejecuta sin romperse.
- Entrenamiento del modelo corre con nuevas columnas.
- `Predictor.predict_probability` acepta el nuevo esquema sin errores.

---

## Fase 2 — Modelo predictivo v2 + explicabilidad
**Duración estimada:** 2 semanas

### Objetivos
- Pasar de clasificación binaria simple a decisiones más ricas.
- Hacer el modelo explicable para usuario final.

### Entregables
- Refactor de modelo a arquitectura multi-objetivo (dirección + magnitud + horizonte).
- Walk-forward validation.
- Módulo de explicabilidad (`SHAP`) y texto de razones legibles.
- Integración de explicaciones al flujo de recomendaciones.

### Criterio de éxito
- Métricas OOS consistentes por ventana temporal.
- Cada señal importante tiene explicación trazable.

---

## Fase 3 — Risk engine institucional
**Duración estimada:** 2 semanas

### Objetivos
- Subir de gestión por posición a gestión de riesgo de portafolio.

### Entregables
- `risk_engine.py` con:
  - límites de exposición total,
  - concentración por sector,
  - tamaño máximo por activo,
  - matriz de correlación,
  - VaR y stress tests básicos.
- Integración en `execution.py` y `trading_bot.py`.
- Fallback de proveedor LLM en `intelligence.py`.

### Criterio de éxito
- Órdenes que violan límites se bloquean con motivo explícito.
- Dashboard muestra riesgo agregado, no solo por trade.

---

## Fase 4 — Diferenciadores Chile (alpha local)
**Duración estimada:** 3 semanas

### Objetivos
- Construir ventajas competitivas difíciles de replicar por apps retail.

### Entregables
- `afp_tracker.py`: detección de presión compradora/vendedora por flujos/rebalanceos AFP.
- `he_analyzer.py`: parser + clasificación IA de Hechos Esenciales CMF y estimación de impacto.
- `regime_detector.py`: bull/bear/sideways para adaptar parámetros dinámicamente.
- Correlación cobre-acción dinámica en tiempo real.
- Score de liquidez avanzado (impacto de orden estimado).

### Criterio de éxito
- Señales integradas en ranking de oportunidades.
- Alertas diferenciadas por evento material y urgencia.

---

## Fase 5 — Métricas reales, trazabilidad y backtest más fiel
**Duración estimada:** 2 semanas

### Objetivos
- Medición correcta de performance y precisión predictiva.

### Entregables
- DB extendida: `nav_history`, `predictions`, `context_history`.
- `stats.py` con ROI, Sharpe, Sortino, Max Drawdown, Profit Factor, alpha vs IPSA.
- Tracker de exactitud de predicciones (predicción vs resultado real a 3 días).
- Backtest alineado al flujo de contexto y features de producción.

### Criterio de éxito
- Métricas no heurísticas, reproducibles y auditables.

---

## Fase 6 — UX de producto (multi-page + perfiles)
**Duración estimada:** 2 semanas

### Objetivos
- Convertir demo en interfaz de producto usable y confiable.

### Entregables
- App Streamlit multi-página:
  - Portafolio,
  - Predicciones,
  - Explainability,
  - Risk,
  - Escenarios,
  - Backtest,
  - AFP flows.
- Perfiles de riesgo (conservador/moderado/agresivo).
- Autenticación básica multi-usuario.

### Criterio de éxito
- Usuario entiende qué recomienda el sistema y por qué.

---

## Fase 7 — Simulación avanzada + puente a broker real
**Duración estimada:** 2 semanas

### Objetivos
- Preparar transición de paper-trading a ejecución real segura.

### Entregables
- `scenario_simulator.py` con shocks predefinidos y custom.
- Abstracción de broker (`BrokerInterface`) separando paper/real.
- Integración inicial con API de broker (cuando se defina proveedor final).

### Criterio de éxito
- Misma lógica de estrategia puede operar en paper o broker real vía adaptador.

---

## Orden recomendado de implementación (secuencial)
1. Fase 1 (base de datos y señales)
2. Fase 3 (riesgo agregado) — en paralelo parcial con Fase 2
3. Fase 2 (modelo v2 + explicabilidad)
4. Fase 5 (métricas y trazabilidad)
5. Fase 4 (diferenciadores Chile)
6. Fase 6 (UX)
7. Fase 7 (broker real)

---

## Riesgos y mitigaciones
- **Dependencia de yfinance / scraping frágil:** implementar cache y fallback.
- **Rate limits LLM:** batch, backoff y proveedor secundario.
- **Data leakage temporal:** validación walk-forward estricta por fecha.
- **Sobreajuste por muchas features:** regularización + selección por importancia + OOS.

---

## Indicadores KPI del proyecto
- Hit-rate y precision de señales BUY.
- Alpha vs IPSA (rolling 3M, 6M, 12M).
- Max drawdown y recuperación.
- % de señales con explicación legible.
- Tiempo promedio desde evento CMF/BCCh hasta alerta.

---

## Estado actual
- [x] Plan maestro documentado.
- [x] Implementación base Fase 1 completada (contexto real + features expandidas + modelo alineado).
- [x] Universo de mercado centralizado y cobertura Chile expandida (IPSA + IGPA/mid-small caps).
- [x] Robustez de market data mejorada (retries y caché macro por ciclo).
- [x] Fase 2 completada: modelo multi-objetivo (dirección + magnitud + horizonte), walk-forward OOS, calibración de threshold y explicabilidad SHAP integrada en señales.
- [x] Fase 3 iniciada: `risk_engine.py` base integrado a flujo de trading (bloqueos por concentración/exposición).
- [x] Fase 3 extendida: VaR paramétrico 1 día y stress tests básicos integrados en logging del ciclo.
- [x] Fase 3 completada: matriz de correlación + VaR por covarianza en ciclo de trading y panel de riesgo agregado en dashboard demo.
- [x] Fase 4 implementada en motor: `afp_tracker.py`, `he_analyzer.py`, `regime_detector.py` integrados al ranking de oportunidades y alertas con urgencia/evento.
- [x] Correlación cobre-acción dinámica y score de liquidez avanzado incorporados en `market_data.py` para señales en tiempo real.
- [x] Fase 5 implementada: DB extendida (`nav_history`, `predictions`, `context_history`) + logging de ciclo + resolución T+3 + `stats.py` institucional (ROI, Sharpe, Sortino, MaxDD, Profit Factor, alpha vs IPSA).
- [x] Fase 6 implementada: app tipo producto con autenticación básica, perfiles de riesgo y secciones Portafolio/Predicciones/Explainability/Risk/Escenarios/Backtest/AFP flows.
- [x] Fase 7 implementada: `scenario_simulator.py` + abstracción `BrokerInterface` con adaptadores paper/real (manual bridge) integrada en bot.
- [x] Reentrenamiento robusto ejecutado (modelo actualizado + `walkforward_results.csv`).
- [x] Guardrail de probabilidad mínima ML para BUY integrado en capa de riesgo IA.
- [ ] Siguiente foco: hardening productivo (tests de regresión, validación histórica extendida y conexión broker real cuando se defina proveedor).
