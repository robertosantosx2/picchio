# Propuesta de adaptación de ideas de TensorFold a Picchio

**Estado:** propuesta de ingeniería; no implica cambios en el motor ni resultados de rendimiento.  
**Fecha:** 2026-11-08  
**Referencia de implementación:** TensorFold `ashhart/TensorFold`, árbol revisado en `main`.  
**Ámbito Picchio:** rama de investigación separada de `feature/cuda-pinned-host-slabs`.

## 1. Resumen ejecutivo

No recomiendo portar TensorFold completo ni reescribir Picchio en Zig. Recomiendo una evolución incremental, conservando el motor C, el backend CUDA opcional, la ruta CPU de referencia y la semántica numérica existente.

La inspección del código de Picchio revela que ya existen varios mecanismos que, a primera vista, podrían parecer ausentes:

- caché de expertos host LRU por capa, expertos fijados y slabs reutilizables;
- lectura paralela de expertos y cargador asíncrono de misses;
- prefetch predictivo de expertos de la siguiente capa, con reserva de slots;
- caché LRU de expertos residentes en VRAM;
- registro opcional de slabs host con CUDA pinned memory.

Por tanto, la oportunidad no está en volver a implementar una caché o un prefetch genérico. Está en cerrar los huecos entre esos subsistemas, medir el coste de cada frontera y mejorar el kernel MoE después de asegurar una ruta de transferencia correcta.

**Recomendación principal:** una ruta de transferencia CUDA asíncrona, acotada y con propiedad explícita de los buffers, precedida por contabilidad unificada y pruebas de ciclo de vida. Después, estudiar la ejecución MoE agrupada de TensorFold. No ampliar todavía el límite pinned de 128 MiB.

## 2. Hallazgos de la comparación

### 2.1 Lo que Picchio ya tiene

| Subsistema | Estado observado en Picchio | Consecuencia |
|---|---|---|
| Lectura de pesos | `expert_load_reusable`, lectura SafeTensors en slabs reutilizables | No duplicar el loader |
| Concurrencia de I/O | `cache_load_batch`, OpenMP y `AsyncLoader` | Mejorar planificación solo si la medición lo justifica |
| Prefetch | `prefetch_issue`, predicción top-k de capa siguiente | Medir precisión, bytes anticipados y contaminación LRU |
| Caché host | Slots por capa, pinned set y LRU | Añadir métricas de presión/evicción, no otra caché |
| Caché GPU | `g_ecache`, LRU por (layer,eid) y presupuesto VRAM | Integrar el presupuesto con el nuevo plan de transferencias |
| Pinned host | `pgpu_host_register/unregister`, límite 128 MiB, fallback pageable | Registro no equivale a transferencia asíncrona |

Referencias Picchio:
- [Loader reutilizable y cache de expertos](https://github.com/robertosantosx2/picchio/blob/feature/cuda-pinned-host-slabs/picchio.c)
- [Backend CUDA y transferencias actuales](https://github.com/robertosantosx2/picchio/blob/feature/cuda-pinned-host-slabs/picchio_cuda.cu)

### 2.2 El hueco más importante: pinned no está conectado a una ruta async

En el backend CUDA actual, `get_expert()` reserva memoria de dispositivo y carga pesos con llamadas `cudaMemcpy(..., cudaMemcpyHostToDevice)`. Después, `pgpu_moe_layer()` ejecuta los kernels de cada experto de forma secuencial y devuelve el resultado con otra copia síncrona.

El registro de slabs host pinned permite que una futura copia asíncrona use memoria apta para DMA, pero el camino actual no llama a `cudaMemcpyAsync`. Por ello:

1. no debemos atribuir una aceleración al registro pinned por sí solo;
2. aumentar el límite pinned sin un consumidor async puede elevar la presión de RAM sin aportar beneficio;
3. la siguiente mejora debe introducir ownership y eventos antes de solapar transferencias.

### 2.3 Qué hace TensorFold que merece adaptación

| TensorFold | Mecanismo estudiado | Adaptación conceptual |
|---|---|---|
| `zig/src/cuda/memory.zig` | Reserva previa, accounting de device + pinned, límites, picos y devolución de reserva ante fallo | Ledger de memoria CUDA en Picchio |
| `zig/src/cuda/copy_source.zig` | Propiedad del origen hasta completar DMA, eventos, buffers reutilizables, cuarentena si no se puede demostrar finalización | Gestor de transferencias async |
| `zig/src/cuda/copy_source_budget.zig` | Límite global de bytes y nodos en custodia, incluyendo recursos retenidos | Presupuesto de staging y transferencias |
| `zig/src/core/kv_ledger.zig` y `kv_pool.zig` | Promesas de memoria antes de admitir trabajo; páginas con referencias y leases | Modelo futuro para KV y memoria de sesiones |
| `zig/src/core/prompt_cache.zig` y `prompt_cache_policy.zig` | Reutilización exacta, borrowers, expiración y eviction segura | Caché futura de prefijos/KV, no prioridad inmediata |
| `zig/src/core/lane_host_admit.zig` y `lane_host_pressure.zig` | Admisión bajo presupuesto y reducción de actividad bajo presión | Política futura de concurrencia |
| `zig/src/families/qwen3_5/cuda_routed.zig` | Router, planificación de filas por experto y kernels grouped decode/prompt | Investigación de kernel MoE agrupado |
| `zig/src/core/lanes/engine.zig`, `drafted.zig`, `shared_drafts.zig` | Planificación de rondas y presupuestado de propuestas especulativas | Inspiración para una futura API de múltiples secuencias |

Fuentes TensorFold:
- [memory.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/cuda/memory.zig)
- [copy_source.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/cuda/copy_source.zig)
- [copy_source_budget.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/cuda/copy_source_budget.zig)
- [kv_ledger.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/core/kv_ledger.zig)
- [kv_pool.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/core/kv_pool.zig)
- [prompt_cache.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/core/prompt_cache.zig)
- [cuda_routed.zig](https://github.com/ashhart/TensorFold/blob/main/zig/src/families/qwen3_5/cuda_routed.zig)
- [kernel contract](https://github.com/ashhart/TensorFold/blob/main/docs/kernel-contract.md)

## 3. Restricciones que gobiernan el diseño

La máquina de referencia es un MSI Thin 15 B12UC con RTX 3050 Laptop de 4 GiB VRAM, aproximadamente 14 GiB RAM visible para el sistema y almacenamiento swap amplio. El presupuesto efectivo de memoria física es mucho más estrecho que el espacio virtual.

Resultados históricos relevantes del equipo:

- CPU baseline: tres pruebas con mediana aproximada de 2.0 tok/s en la configuración registrada.
- CUDA experts actual: aproximadamente 1.6–1.9 tok/s en las pruebas registradas; no superó de forma consistente la ruta CPU.
- Slabs pinned: el diagnóstico de 8 tokens llegó a 127.3 MiB de 128 MiB; 1.277 rechazos por límite y cero fallos CUDA/alineación/contabilidad.
- La inferencia pinned experimental de 8 tokens no produjo reutilización efectiva y fue más lenta que el baseline en algunas ejecuciones. Esos ensayos cortos no son un benchmark de rendimiento.

Implicaciones:

- mantener el límite pinned de 128 MiB hasta caracterizar tamaños, simultaneidad y presión de RAM;
- no confundir RAM virtual/swap con RAM física libre;
- no aumentar la caché VRAM sin contar pesos, scratch, buffers de activación, contexto CUDA y margen del escritorio;
- preservar una ruta CPU y una ruta CUDA síncrona de referencia;
- toda mejora debe medirse con la misma secuencia, prompt, contexto, tokens y estado de caché.

## 4. Arquitectura propuesta

### 4.1 Ledger de memoria CUDA (primera pieza)

Introducir contabilidad explícita, independiente de la política de colocación:

- bytes host pinned registrados;
- bytes host staging asignados y pendientes de liberar;
- bytes device de expertos residentes;
- bytes device de scratch;
- bytes de transferencias en vuelo;
- picos por categoría y total;
- reservas pendientes antes de llamar a CUDA;
- errores separados: límite, alineación, CUDA y bookkeeping.

La reserva debe ocurrir antes de la asignación. Si falla `cudaHostRegister`, `cudaMalloc` o la preparación del evento, se devuelve exactamente la reserva que no llegó a convertirse en recurso vivo. Los recursos que no se pueden liberar de forma segura siguen contados como retenidos; nunca se descuentan solo porque se perdió la referencia.

No usar un único contador ambiguo para memoria pinned registrada, buffers pageable y buffers host de staging. Registrar una vista consolidada, pero conservar las categorías.

### 4.2 Gestor de transferencias host → device (segunda pieza)

Crear una capa pequeña en el backend CUDA con un contrato similar a:

- `stage_acquire(bytes)`: obtiene un buffer pinned del pool o reserva uno nuevo dentro del presupuesto;
- `copy_submit(dst, src, bytes, stream)`: encola DMA y registra evento;
- `copy_poll(ticket)` / `copy_wait(ticket)`: confirma finalización;
- `stage_release(ticket)`: solo devuelve el buffer al pool después de completar el evento;
- `stage_quarantine(ticket)`: conserva el buffer si un error impide demostrar que el DMA terminó.

En la primera versión usar un único stream de transferencia y un pool pequeño, acotado. Evitar todavía un sistema multistream complejo.

Una segunda fase puede separar stream de copia y stream de cómputo:
1. encolar copia del experto siguiente;
2. registrar evento tras esa copia;
3. hacer que el stream de cómputo espere el evento antes de consumir ese experto;
4. ejecutar el experto actual mientras se transfiere el siguiente;
5. no permitir eviction de un DevExpert con copia o kernel en vuelo.

La vida del slab de Picchio debe extenderse hasta que el evento confirme la finalización. El worker de I/O no debe poder reciclar o sobrescribir ese slab antes de ese punto. El cierre debe drenar streams y eventos antes de destruir buffers o contexto.

### 4.3 Integración con el LRU de VRAM

El estado de un experto GPU no debería ser solo presente/ausente. Propuesta de estados:

- EMPTY;
- ALLOCATED;
- UPLOADING;
- RESIDENT;
- IN_USE;
- EVICT_PENDING;
- FAILED.

La eviction solo puede seleccionar un experto RESIDENT sin referencias activas ni operaciones pendientes. La selección top-k de una capa debe proteger todas sus entradas antes de subir los misses, como ya hace Picchio al proteger expertos residentes del mismo lote.

No crear un segundo LRU. Evolucionar `g_ecache` y su política existente para que entienda tickets, eventos y bytes reservados.

### 4.4 Kernel MoE agrupado (tercera pieza)

TensorFold separa la preparación de rutas y miembros de expertos del kernel de cómputo, y dispone de rutas grouped para decode y prompt. Picchio, en cambio, ejecuta para cada experto seleccionado una secuencia de kernels gate/up, SwiGLU, down y acumulación.

Propuesta de investigación:

- construir primero un microbenchmark aislado para INT4 gs64;
- preparar una lista compacta de pares (token, experto, peso);
- agrupar filas por experto;
- calcular gate/up y down para el grupo;
- combinar resultados conservando el orden y los pesos de router;
- comparar decode de una fila y prefill por lotes por separado.

No asumir que un kernel grouped es automáticamente mejor en decode de un solo token. Puede reducir lanzamientos en prefill, pero añadir trabajo de planificación y scratch. En una RTX 3050, la ocupación, el ancho de memoria y el tamaño real del lote determinarán el resultado.

INT3 se abordará solo después de validar INT4, porque su empaquetado bit-plane y su contrato numérico complican el primer prototipo.

## 5. Plan de trabajo por fases

| Fase | Entregable | Prueba de aceptación | Riesgo |
|---|---|---|---|
| 0. Baseline reproducible | Script/protocolo con CPU, CUDA síncrono, cache fría/caliente y métricas | Repetición estable; conservar logs completos | Bajo |
| 1. Ledger | Contadores y reservas para host pinned, staging y VRAM | Tests de límite exacto, overflow, rollback y cierre | Bajo-medio |
| 2. Transferencia async simple | Pool acotado + un stream + eventos + fallback síncrono | Tests de vida útil, error y equivalencia numérica | Medio |
| 3. Overlap copy/compute | Segundo stream y tickets por experto | Evidencia de solapamiento real en timeline y menor tiempo total | Alto |
| 4. Grouped MoE INT4 | Microbenchmark y backend experimental | Mismos tokens dentro del contrato numérico; mejora reproducible | Alto |
| 5. INT3 y prefill grouped | Segunda ruta de kernel | Ganancia por separado en decode y prefill | Alto |
| 6. KV/prompt cache | Solo si se incorpora servicio multi-sesión/contextos largos | Admisión sin OOM, restore exacto y eviction segura | Alto |
| 7. Speculative decode | Copias verificadas o drafter pequeño | Igualdad de tokens aceptados y mejora neta incluyendo overhead | Alto |

### Criterios de promoción

Un cambio pasa a ser candidato a la ruta predeterminada solo si:

1. supera tests de corrección y de ciclo de vida;
2. no modifica el resultado CPU de referencia;
3. no produce errores CUDA ni fugas de recursos;
4. mejora el tiempo total de generación, no solo el tiempo de copia;
5. la mejora se repite en al menos cinco ejecuciones comparables y se informa mediana y dispersión;
6. no dispara RAM, swap o VRAM hasta un nivel que comprometa el escritorio;
7. se conserva un mecanismo sencillo para desactivarlo.

Si no mejora el tiempo total, se conserva como opción experimental o se descarta.

## 6. Protocolo de benchmark

Cada ensayo debe registrar:

- SHA de Picchio, versión del compilador y flags;
- modelo, revisión, cuantización, número de expertos y top-k;
- prompt exacto, contexto y tokens solicitados/generados;
- estado frío/caliente de caché host y VRAM;
- latencia de prefill, tiempo de decode y tok/s;
- hits/misses/evictions de caché host y GPU;
- bytes leídos de disco y tiempo de I/O;
- bytes pinned solicitados, registrados, rechazados y retenidos;
- bytes subidos a VRAM, número de copias, tiempo de DMA y tiempo de kernel;
- pico RAM, swap y VRAM;
- temperatura, potencia y clocks GPU;
- resultado/token diff frente al baseline.

Comparaciones obligatorias:
- CPU baseline;
- CUDA síncrono actual;
- CUDA async con el mismo presupuesto;
- CUDA async con overlap, solo después de validar la fase anterior.

No mezclar cache fría y caliente ni comparar prompts distintos. Las pruebas de 8 tokens sirven para depuración, no para concluir rendimiento.

## 7. Qué NO recomiendo portar ahora

1. **KV paginada:** el foco actual es inferencia MoE de una secuencia y el problema medido está en carga/transferencia de expertos. No añadir un pool KV complejo antes de necesitar concurrencia o contextos mayores.
2. **Prompt cache completa:** exige snapshots correctos de todo el estado del modelo, reglas exactas de prefijo y control de vida de los borrowers. Es una inversión posterior.
3. **MTP/drafter y speculative trees:** añaden memoria, kernels y coste de verificación. Primero hacer que el camino normal sea eficiente.
4. **Planificador multi-lane completo:** no es necesario para el objetivo actual y podría complicar el código C.
5. **Aumentar pinned a 192/256/384 MiB ahora:** no hay distribución de tamaños ni prueba que demuestre que esos límites son seguros o útiles.
6. **Copiar código fuente de TensorFold directamente:** adaptar contratos e ideas; cualquier trasplante concreto requiere revisar licencia y avisos.

## 8. Riesgos y mitigaciones

- **Uso prematuro de un slab:** ownership por ticket y evento; no liberar ni reciclar hasta finalización confirmada.
- **Eviction durante DMA/kernel:** estados explícitos y referencias activas por DevExpert.
- **Fallo de evento o sincronización:** cuarentena; conservar bytes en el ledger hasta recuperación segura.
- **Presión RAM:** presupuesto total separado para slabs pinned, staging y memoria pageable; fallback síncrono/pageable.
- **Deadlocks:** no esperar a un worker manteniendo locks que el worker necesite; pruebas de shutdown y error.
- **Cambios numéricos:** no alterar formatos ni orden de suma en la fase de transferencia.
- **Falsa mejora:** medir end-to-end, con cinco o más repeticiones y cache state controlado.

## 9. Licencias y reutilización

TensorFold se distribuye bajo Apache License 2.0; Picchio declara MIT. Las ideas y los patrones arquitectónicos pueden implementarse de forma independiente. Si se copia o adapta código de TensorFold, habrá que conservar los avisos aplicables, incluir la licencia y cumplir las obligaciones de Apache-2.0, incluida la identificación de cambios en archivos modificados. Esta propuesta no incorpora código fuente de TensorFold.

## 10. Decisión recomendada

**Aprobar como siguiente trabajo técnico las fases 0 y 1, y diseñar la fase 2 sin activarla todavía.**

En concreto:

1. cerrar primero la instrumentación pinned ya iniciada y caracterizar tamaños/bytes rechazados;
2. añadir un ledger con reservas y contadores completos;
3. diseñar un pool de staging pequeño, con un único stream y eventos;
4. construir pruebas de ownership, cancelación, error y shutdown;
5. solo entonces probar `cudaMemcpyAsync` y medir si mejora el tiempo de generación;
6. investigar grouped INT4 como línea paralela de microbenchmark, sin reemplazar todavía el kernel actual.

La primera meta no es prometer más tokens por segundo. Es eliminar incertidumbre sobre dónde se gasta el tiempo y garantizar que cualquier solapamiento CUDA sea seguro. Después podremos decidir, con evidencia, qué ruta merece optimización.

---

## Apéndice: archivos Picchio que probablemente cambiarían

- `picchio_cuda.cu`: ledger, staging pool, tickets/eventos, estados del LRU GPU y, más adelante, grouped kernels.
- `picchio_cuda.h`: API mínima para inicializar, consultar y drenar el subsistema de transferencias, si se necesita exponerla al motor C.
- `picchio.c`: solo si hace falta coordinar el ciclo de vida de slabs con tickets de transferencia; evitar mezclar política CUDA en el loader.
- `Makefile` y CI: tests CPU/sintéticos y build CUDA separado cuando haya runner CUDA disponible.
- `docs/`: protocolo de benchmark, matriz de pruebas y resultados reales.

La ruta CPU de referencia y la biblioteca CUDA funcional existente deben permanecer intactas mientras se desarrolla y valida la implementación experimental.
