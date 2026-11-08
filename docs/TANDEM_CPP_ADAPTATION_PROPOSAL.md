# tandem.cpp: análisis y mejoras aplicables a Picchio

**Estado:** investigación y propuesta; no cambia el motor ni activa nuevas rutas de inferencia.  
**Fecha:** 2026-11-08  
**Proyecto estudiado:** [ewhacc/tandem.cpp](https://github.com/ewhacc/tandem.cpp)  
**Base del parche:** llama.cpp `f5222f25c` (2026-10-28).  
**Rama de investigación Picchio:** `research/tensorfold-adaptation`.

## 1. Conclusión ejecutiva

La aportación central de tandem.cpp no es simplemente una caché LRU de expertos. Es una política de **ejecución híbrida por experto**: para cada capa, los expertos que ya están en VRAM los calcula la GPU; parte de los expertos que faltan se suben a la GPU; el resto se calcula en CPU en paralelo con la ejecución GPU. La proporción CPU/GPU se adapta a las velocidades medidas del procesador y del enlace host→device.

Hay dos ideas más que merecen atención especial:

1. **Prefetch real hacia VRAM en un stream lateral**, con eventos y espera diferida solo cuando un kernel va a leer un slot que aún se está llenando.
2. **CPU MoE agrupada por experto y por bloques de filas**, reutilizando la entrada cuantizada y los pesos del experto para todas las parejas token-experto que lo necesitan.

No recomiendo copiar el parche entero. Es un parche de 432 KB que modifica 35 archivos de llama.cpp y concentra una gran cantidad de lógica en el subsistema MoE. La implementación está muy adaptada a GGUF, ggml y a las arquitecturas ensayadas.

Para el MSI RTX 3050 de 4 GiB, mi orden recomendado es diferente del orden de las máquinas de los benchmarks de tandem:

1. Instrumentar el coste de misses y medir por separado disco, CPU y PCIe.
2. Prototipar **prefetch GPU de expertos de la siguiente capa**, con eventos y un stream lateral, sin cambiar la matemática.
3. Probar un **split CPU/GPU pequeño y adaptativo**, solo con expertos que ya estén disponibles en RAM.
4. Mejorar el kernel CPU INT4/INT3 por bloques y por experto, si las medidas justifican esa ruta.
5. Dejar doorbell, prefill híbrido avanzado y MTP para una fase posterior.

La principal limitación de nuestro equipo es la memoria: el modelo Qwen3-30B-A3B INT4 ocupa aproximadamente 20,6 GB, mientras que el sistema muestra unos 14 GiB de RAM y dispone de 4 GiB de VRAM. En las pruebas anteriores hubo miles de lecturas de disco y presión de memoria. tandem.cpp reconoce expresamente que su ruta CPU no resuelve los misses que llegan desde SSD; en un equipo donde el conjunto de expertos no cabe en RAM, esa limitación es decisiva.

## 2. Alcance real del estudio

El repositorio de tandem.cpp contiene README, script de construcción, scripts/datos de benchmark y un parche de aproximadamente 432.439 bytes. El parche modifica 35 archivos. El bloque más grande es `src/llama-moe-cache.cpp`, con unas 2.577 líneas añadidas y 127 eliminadas. Se revisaron la arquitectura del parche, el código central de caché/híbrido, el kernel CPU, los cambios de grafo/backend CUDA, las opciones de ejecución, los benchmarks y las limitaciones publicadas.

Esto es un análisis de los subsistemas relevantes, no una afirmación de que se haya auditado línea por línea todo el código de llama.cpp que el parche modifica.

Referencias principales:
- [Repositorio y README](https://github.com/ewhacc/tandem.cpp)
- [Parche completo](https://github.com/ewhacc/tandem.cpp/blob/main/tandem.patch)
- [Código de caché MoE del parche](https://github.com/ewhacc/tandem.cpp/blob/main/tandem.patch) — buscar `src/llama-moe-cache.cpp`.
- [Benchmarks](https://github.com/ewhacc/tandem.cpp/tree/main/bench)
- [Commit base llama.cpp](https://github.com/ggml-org/llama.cpp/commit/f5222f25c)

## 3. Cómo funciona tandem.cpp

### 3.1 La decisión por capa

Para cada capa MoE, tandem obtiene los expertos seleccionados y los separa en:

- hits: expertos que ya residen en la caché GPU;
- misses admitidos: expertos ausentes que se cargan a VRAM;
- misses CPU: expertos ausentes que se calculan desde los pesos residentes en memoria host.

La caché LRU no admite todos los misses. `plan_hybrid()` cuenta cuántos tokens de la microbatch usan cada experto, prioriza los expertos más compartidos y admite solo una fracción de los misses. Los expertos elegidos por CPU apuntan a un slot GPU cero para que la ruta GPU no aporte nada por ellos. Después se suma el resultado CPU.

Esto es importante: no es una política de offload por capa completa. La unidad de planificación es el experto seleccionado en esa capa.

### 3.2 Regla adaptativa CPU/PCIe

Tandem mide el ancho de banda de subida host→device con copias de expertos y mide la velocidad efectiva de la ruta CPU sobre los misses. Usa una regla derivada de Strata:

\[
s = \frac{L}{L + kR}
\]

donde:
- (L) es el ancho de banda host→device medido;
- (R) es la tasa de cómputo CPU medida en bytes de experto por tiempo;
- (k) es un factor empírico (1,85 por defecto);
- (s) es la fracción de misses que se sube a GPU; el resto se calcula en CPU.

El resultado se cuantiza en pasos de 5 puntos porcentuales y se limita a un rango conservador. También hay un controlador basado en espera/ociosidad de CPU. El README y los comentarios del código explican que el controlador por espera puede pasarse de CPU porque las subidas PCIe y el cálculo CPU compiten por el mismo ancho de banda de RAM.

Además, tandem incorpora un **A/B gate**: mide ventanas con la ruta CPU activada y desactivada, compara la latencia de la microbatch y conserva la ruta solo si gana con margen. Esto evita mantener una ruta híbrida cuando su coste fijo supera el ahorro.

**Lección para Picchio:** la proporción no debe elegirse por intuición ni fijarse universalmente. Hay que medir CPU, PCIe, RAM y tiempo total; después debe existir un mecanismo automático para apagar el híbrido cuando no compensa.

### 3.3 Kernel CPU especializado

La ruta CPU no llama simplemente al mismo bucle de expertos secuencial. Agrupa las parejas (token, experto), ordena el trabajo por experto y reparte bloques de filas entre trabajadores.

El código:
- cuantiza una vez la entrada de cada token para el tipo que espera el producto matricial;
- reutiliza esa entrada para los expertos del token;
- agrupa las parejas de un mismo experto, de modo que las filas de sus pesos se recorren de forma contigua;
- divide gate/up y down en bloques de filas;
- permite una ruta de productos multi-token para determinados formatos IQ;
- incorpora una ruta SIMD específica para Q2_0 con AVX2/AVX-VNNI;
- mantiene el orden de acumulación de los pares cuando usa el camino de down por experto.

La ruta SIMD Q2_0 no se puede copiar directamente a Picchio: nuestro formato es INT4/INT3 group-scaled gs64, no Q2_0 GGUF. Lo transferible es el diseño de trabajo, la reutilización de activaciones cuantizadas y el paralelismo por bloques.

### 3.4 Prefetch GPU en un stream lateral

Tandem puede aplicar el router de la capa siguiente a una representación proxy de la entrada actual y predecir expertos probables. Un hilo prepara y sube esos expertos mediante un backend/stream lateral.

La característica importante es el **lazy join**:
- el prefetch no obliga a la GPU principal a esperar al terminar cada capa;
- se registra qué slots están siendo escritos;
- la GPU espera el evento solo cuando una operación va a leer un slot pendiente o cuando necesita reutilizarlo;
- los prefetches cancelados o inútiles se cuentan;
- un slot no se puede reutilizar mientras la copia pendiente pueda escribirlo.

Es una optimización de latencia de misses, no un cambio de resultado: una predicción errónea desperdicia una subida o una plaza temporal de caché, pero no cambia los expertos que el router selecciona.

### 3.5 Doorbell y suma desde memoria host pinned

El modo doorbell es una integración más profunda con ggml:
- un kernel GPU publica IDs, entrada y pesos de router en un bloque de control host pinned;
- un hilo host observa la petición, planifica LRU, lanza el trabajo CPU y organiza las subidas;
- el kernel GPU espera la respuesta y continúa;
- `ggml_host_add` permite que la GPU lea la salida CPU desde memoria host pinned después de una señal.

Esto elimina algunos callbacks/sincronizaciones del camino convencional, pero requiere nuevos operadores ggml, kernels CUDA, protocolo de memoria compartida, atomics, fences, gestión de timeouts y shutdown. No es una primera adaptación razonable para Picchio.

### 3.6 Prefill y otras optimizaciones

El parche incluye una ruta CPU/GPU separada para microbatches grandes, predicción lookahead, pre-read de pesos, afinidad de hilos CPU, empaquetado de inputs del scheduler, CUDA graphs por forma, caché de activaciones Q8 y mejoras de MTP/speculative decode.

No todas son transferibles:
- la ruta de prefill híbrida presupone el grafo y callbacks de ggml;
- los CUDA graphs por forma y packed inputs son optimizaciones del scheduler de llama.cpp;
- el MTP del ejemplo usa un head externo Q4_K específico de Qwen3.8;
- las fusiones GDN/HC y kernels Qwen3.8 son específicos de esa arquitectura;
- las optimizaciones Q2_0/IQ son específicas de cuantizaciones GGUF.

## 4. Comparación con Picchio actual

| Área | Picchio actual | Aportación tandem | Aplicabilidad |
|---|---|---|---|
| Caché host de expertos | LRU/slots, expertos fijados y slabs reutilizables | Cuenta frecuencia de uso por microbatch para decidir admisiones | Media; útil en lotes, menos en decode de un token |
| Lectura de expertos | `pread`/SafeTensors, lecturas paralelas y AsyncLoader | CPU ejecuta misses mientras GPU ejecuta hits | Alta como concepto, condicionada por RAM |
| Prefetch siguiente capa | Predicción proxy y `prefetch_issue` a host | Prefetch de pesos directamente a VRAM en stream lateral | Muy alta |
| Caché VRAM | LRU por (layer,eid), carga síncrona | Estados de subida pendiente y lazy event wait | Muy alta |
| CPU MoE | `expert_apply` secuencial, cuantización INT4/INT3 gs64 | Pool de hilos, agrupación por experto y bloques de filas | Alta para experimentar |
| Split CPU/GPU | La llamada CUDA intenta calcular la capa completa; si falla, fallback CPU | Reparto explícito de misses entre CPU y GPU | Alta en teoría; riesgo alto por RAM y CPU |
| Pinned host | Registro de slabs, límite 128 MiB, fallback | Buffers pinned usados por DMA y operadores de host | Alta, pero requiere ciclo de vida seguro |
| Doorbell | No existe scheduler de grafos equivalente | Operador GPU/host cooperativo | Baja ahora |
| MTP/speculación | Hay experimentos de self-draft/otros mecanismos en Picchio | MTP con head cuantizado y catch-up diferido | Condicional; no trasladar sin modelo/head compatible |

Archivos Picchio relevantes:
- [picchio.c — carga, caché, prefetch y moe_forward](https://github.com/robertosantosx2/picchio/blob/feature/cuda-pinned-host-slabs/picchio.c)
- [picchio_cuda.cu — caché VRAM y kernels INT4/INT3](https://github.com/robertosantosx2/picchio/blob/feature/cuda-pinned-host-slabs/picchio_cuda.cu)
- [picchio_cuda.h — interfaz CPU/CUDA](https://github.com/robertosantosx2/picchio/blob/feature/cuda-pinned-host-slabs/picchio_cuda.h)

## 5. Mejoras propuestas para Picchio, por prioridad

### P0 — Instrumentación y baseline específico de misses

Antes de modificar la ruta, registrar por capa/token:
- expertos seleccionados;
- hits VRAM, misses VRAM y expertos ya disponibles en host;
- bytes leídos de disco, tiempo de lectura y espera del loader;
- bytes transferidos a GPU y duración de subida;
- tiempo CPU por experto y por capa;
- tiempo de kernel GPU;
- tiempo de espera entre loader, CPU y GPU;
- presión de RAM, swap y VRAM;
- número de evictions, uploads fallidos y prefetched-but-unused.

Separar explícitamente tres tipos de miss:
1. **VRAM miss / host hit:** pesos ya disponibles en RAM; candidato ideal para CPU/GPU split o subida async.
2. **Host cache miss / page-cache hit:** se puede leer desde caché del SO, pero no está en el slot de Picchio.
3. **SSD miss:** lectura física o page fault con coste alto; no asignarlo automáticamente a CPU, porque puede bloquear el trabajo y degradar todo el sistema.

La métrica que debe guiar la decisión es el tiempo end-to-end, no el número de bytes pinned ni el porcentaje de hits aislado.

### P1 — Prefetch de expertos a VRAM con stream lateral

**Es la mejora de tandem que priorizaría para el MSI.**

Picchio ya predice expertos de la capa siguiente y los precarga en la caché host. El siguiente paso natural es que el backend CUDA pueda recibir una petición de prefetch de esos expertos y subirlos a VRAM mientras se calcula la capa actual.

API experimental posible:
- `pgpu_prefetch_expert(layer, eid, host_weights, ...)`;
- `pgpu_prefetch_poll(ticket)`;
- `pgpu_moe_layer(...)` consume un experto RESIDENT o espera el evento de uno UPLOADING;
- `pgpu_drain_prefetch()` antes del shutdown.

Estados mínimos del experto GPU:
`EMPTY → ALLOCATED → UPLOADING → RESIDENT → IN_USE → EVICT_PENDING`.

Condiciones de seguridad:
- reservar bytes de VRAM antes de `cudaMalloc`;
- no publicar un slot como RESIDENT hasta que el evento de copia complete;
- no reciclar el slab host hasta completar el DMA;
- no evictar un experto con una copia o kernel pendiente;
- cancelar prefetches obsoletos sin liberar buffers en vuelo;
- limitar bytes y número de subidas concurrentes;
- fallback a la ruta síncrona actual si no hay recursos o falla un evento.

Comenzar con un único stream de transferencia, un pequeño número de tickets y una sola capa de lookahead. No implementar todavía doorbell ni varios streams de cómputo.

**Criterio de éxito:** reducir la espera por misses VRAM y el tiempo de generación total, sin aumentar lecturas de disco ni presión de memoria. Medir también porcentaje de prefetch útil.

### P2 — Split CPU/GPU adaptativo de expertos ausentes

La siguiente línea experimental es separar los misses de VRAM:
- los hits residentes permanecen en GPU;
- una parte de los misses se sube a GPU;
- los misses restantes se calculan en CPU;
- se suman ambas contribuciones conservando el peso de router y el contrato numérico.

En Picchio puede ser más sencillo que el diseño de ggml: el motor ya tiene una función CPU `expert_apply` y una interfaz CUDA `pgpu_moe_layer`. Se podría construir una lista CPU y otra GPU, iniciar un worker CPU con la primera y llamar al backend CUDA con la segunda. Al terminar, se combina la salida CPU con la salida GPU.

Sin embargo, hay tres riesgos importantes:

1. **RAM insuficiente:** el CPU solo debe calcular expertos cuyos pesos estén realmente disponibles en RAM. No debe convertir misses SSD en trabajo CPU.
2. **CPU posiblemente más lenta que PCIe:** el baseline CPU ronda 2 tok/s y el backend CUDA actual 1,6–1,9 tok/s en las pruebas históricas. Esto justifica medir el solapamiento, no asumir que CPU es mejor.
3. **Diferencias numéricas:** separar la suma CPU/GPU cambia el orden de acumulación y puede cambiar redondeos. Para comparar con máxima precisión, el prototipo debería devolver contribuciones por experto y reducirlas en el orden top-k original; si se usa una suma parcial CPU, fijar una tolerancia y comprobar tokens.

El reparto inicial debe ser conservador y configurable. El auto-control puede usar la regla (L/(L+kR)), pero debe incorporar un A/B gate que desactive el CPU path si no mejora la latencia por microbatch.

No activar `tail skip` de tandem: omitir expertos cambia el resultado del modelo y no es una optimización exacta.

### P3 — Kernel CPU INT4/INT3 por experto y bloques de filas

El kernel CPU actual procesa los expertos seleccionados secuencialmente. Probar un pool acotado que:
- agrupe pares (token, experto);
- recorra pesos de un experto una vez para todos los tokens que lo usan;
- divida gate/up y down en bloques de filas;
- cuantice la entrada una vez por token y reutilice esa representación;
- evite que varios hilos escriban la misma región de salida;
- mantenga el orden de suma por experto y peso de router.

La ruta tandem Q2_0 AVX2/VNNI no es portable directamente al INT4/INT3 gs64 de Picchio. Primero hay que perfilar `matmul_qt` y los kernels INT4/INT3 actuales para saber si la CPU está limitada por cómputo, ancho de banda o falta de paralelismo.

Para el MSI, empezar con 2 hilos CPU de cálculo además del hilo coordinador y comparar 2/4/6 trabajadores. No reservar todos los 16 hilos lógicos: hay I/O, atención, SO y otras tareas compitiendo.

### P4 — Pre-read de pesos mientras CPU/GPU trabajan

Tandem tiene una cola de pre-read para traer a RAM chunks de expertos probables durante los intervalos en que el CPU está ocupado. Picchio ya tiene AsyncLoader y prefetch host. Solo añadiría una cola independiente si las métricas demuestran que:
- la GPU/CPU dejan RAM/SSD ociosa;
- la predicción siguiente capa tiene precisión suficiente;
- el pre-read no desplaza pesos calientes de la page cache;
- no aumenta el swap ni la latencia de los misses reales.

Es especialmente importante separar pre-read SSD de subida PCIe. No se debe alimentar más trabajo de disco si el loader ya está saturado.

### P5 — Política de caché por frecuencia de uso del microbatch

Tandem elige para la caché los expertos que sirven a más tokens de la microbatch. En decode de un token todos los expertos seleccionados tienen frecuencia uno, por lo que esta política apenas aporta. En prefill o lotes con varios tokens sí puede reducir uploads y proteger expertos compartidos.

Picchio ya tiene `eusage`, `eheat`, promoción y predicción. Recomendación: medir frecuencia por batch y comparar con el LRU/heat actual antes de cambiar la política global.

### P6 — Doorbell / host_add y prefill híbrido

No lo recomiendo para la primera implementación. Exige:
- un protocolo de control host/device con atomics y fences;
- nuevos kernels CUDA y primitivas de backend;
- un bloque pinned con IDs, activaciones, pesos, slots y flags;
- gestión de timeout, cancelación, shutdown y recuperación;
- una ruta de prefill distinta a decode.

La idea podría ser útil si el backend evoluciona a un scheduler de grafos, pero Picchio es actualmente un motor C con una llamada CUDA por capa. El coste de introducir una infraestructura similar a ggml sería desproporcionado.

### P7 — MTP / speculative decode

Tandem reduce el tamaño de un head MTP concreto de 3,6 a 1,8 GiB cuantizándolo a Q4_K. No se puede trasladar sin disponer de un drafter compatible y sus pesos. En la RTX 3050 de 4 GiB, cualquier head adicional compite directamente con la caché de expertos y el scratch. Mantener la línea de especulación de Picchio separada y medir aceptación, VRAM y tokens/s netos.

## 6. Por qué no copiar directamente los resultados publicados

Tandem publica 97,3 tok/s en Qwen3.8-Flash-Next IQ3_S + MTP en una RTX 4090 de 24 GiB con 64 GB DDR4 y un i5-12400F. El mismo README presenta 51,5 tok/s en gpt-oss-120b y pérdidas de rendimiento cuando los expertos tienen que venir desde SSD. También declara pruebas CUDA en sm_89 y sm_120; la RTX 3050 de este proyecto es sm_86 y no figura entre los targets ensayados.

Esos resultados son mediciones comunicadas por los autores, no una predicción para el MSI. Cambian el modelo, la cuantización, la GPU, el ancho de banda RAM/PCIe, el contexto y la disponibilidad de pesos en memoria.

El README reconoce:
- menor ganancia cuando el modelo casi cabe en VRAM;
- menor beneficio con peticiones concurrentes y cachés pequeñas;
- fallo de la estrategia cuando el modelo supera mucho la RAM y los misses llegan desde SSD;
- límites de soporte CUDA/Linux y arquitecturas probadas.

Fuentes:
- [README: benchmark, configuración y limitaciones](https://github.com/ewhacc/tandem.cpp/blob/main/README.md)
- [Benchmarks de tandem.cpp](https://github.com/ewhacc/tandem.cpp/tree/main/bench)
- [Parche y base exacta](https://github.com/ewhacc/tandem.cpp)

## 7. Plan de experimentos propuesto

| Experimento | Cambio | Qué responde | Decisión |
|---|---|---|---|
| T0 | Solo instrumentación | ¿Cuánto tiempo es SSD, CPU, PCIe, GPU y espera? | Elegir la ruta |
| T1 | Prefetch GPU de una capa, un stream | ¿Se solapan las subidas con el cómputo actual? | Seguir o descartar P1 |
| T2 | Split CPU/GPU fijo 10%, 20%, 30% de misses host-resident | ¿Puede el CPU ayudar mientras la GPU calcula? | Hallar zona útil |
| T3 | Auto-share por L/R | ¿La proporción adaptativa supera al mejor fijo? | Activar solo si gana |
| T4 | A/B gate CPU path ON/OFF | ¿El coste CPU compensa en este modelo/caché? | Apagar automáticamente si no |
| T5 | CPU worker pool INT4 | ¿El CPU puede procesar más expertos con menor latencia? | Afinar hilos y bloques |
| T6 | Prefill batch con frecuencia por experto | ¿La admisión por frecuencia mejora los lotes? | Política separada decode/prefill |

Para T0–T4 usar la misma secuencia de tokens, prompt, contexto y estado de caché. Al menos cinco repeticiones comparables por variante, mediana y dispersión. Separar pruebas de caché fría y caliente. No comparar solo tok/s: registrar TTFT, prefill tok/s, decode tok/s, tiempo por capa, CPU, VRAM, RAM, swap y lecturas de disco.

Criterios para aceptar una mejora:
- sin OOM, deadlock, corrupción o liberación prematura;
- todas las rutas exactas calculan todos los expertos seleccionados;
- tokens iguales cuando se exige igualdad estricta, o tolerancia numérica documentada cuando cambia el orden de suma;
- mejora end-to-end repetible, no solo una copia o kernel aislado;
- consumo de RAM/VRAM dentro de márgenes que permitan usar Ubuntu con normalidad;
- ruta antigua disponible como fallback.

## 8. Diseño de implementación recomendado

### Etapa A: no tocar la matemática

Primero ampliar la instrumentación y añadir al backend CUDA:
- consulta de residencia por experto;
- estado UPLOADING separado de RESIDENT;
- reserva de bytes y protección frente a eviction;
- stream lateral de transferencia;
- evento por upload;
- API de prefetch con fallback síncrono;
- drenaje de tickets al apagar.

El loader C conserva la propiedad del slab hasta el evento. La política LRU no ve el experto como disponible antes de completar la subida.

### Etapa B: híbrido CPU/GPU detrás de una bandera

Añadir una bandera experimental `GPU_HYBRID=1` y un porcentaje fijo de CPU/GPU. No activar auto al principio. Solo se pueden asignar a CPU expertos que tengan pesos válidos en RAM y no estén pendientes de I/O. La salida CPU se calcula en un worker y se suma después de la GPU.

Empezar por una capa sintética o microbenchmark de expertos, y después por inferencia real con top-k. Comparar salida por experto frente al camino CPU original. Mantener desactivado en los modos oracle/trace hasta tener cobertura numérica.

### Etapa C: auto-share y A/B gate

Cuando el split fijo funcione, medir:
- bytes de experto / tiempo de CPU por experto;
- bytes H2D / tiempo de subida;
- tiempo CPU que queda ocioso;
- tiempo que la GPU espera a CPU;
- tiempo total de la capa y del token.

Calcular una proporción inicial por L/R y ajustarla en pasos pequeños. Añadir A/B gate con ventanas suficientemente largas y comparables. No adaptar a partir de un único token ni de muestras contaminadas por misses SSD.

### Etapa D: kernel CPU por bloques

Solo después de medir el split. Implementar pool acotado y kernels especializados para el formato real de Picchio (INT4/INT3 gs64), no portar Q2_0. Verificar que la cuantización de activaciones se reutiliza y que el paralelismo no crea escrituras concurrentes sobre la misma salida.

## 9. Riesgos técnicos que no deben ocultarse

- **RAM compartida:** CPU y DMA compiten por el mismo controlador de memoria. Más CPU no significa automáticamente más throughput.
- **Swap/SSD:** si un experto no está residente, el CPU path puede convertir un miss GPU en un miss de almacenamiento mucho más caro.
- **CPU i7-12650H:** tiene 16 hilos lógicos, pero no equivale al ancho de banda de memoria de los equipos DDR4 de 8 canales o DDR5 de servidor del benchmark.
- **RTX 3050 4 GiB:** la caché GPU es pequeña; demasiados prefetches pueden expulsar expertos útiles.
- **sm_86:** tandem declara ensayos en sm_89/sm_120; cualquier kernel o ruta nueva debe compilarse y validarse explícitamente en sm_86.
- **Redondeo CPU/GPU:** las rutas pueden no ser bit-exactas aunque calculen todos los expertos.
- **Thread oversubscription:** el worker de CPU no debe competir con demasiados hilos de I/O, router, atención o el escritorio.
- **Shutdown y errores CUDA:** no liberar buffers pinned, eventos ni slabs mientras una copia o kernel pueda seguir usándolos.

## 10. Decisión

**Priorizar P1: prefetch a VRAM con stream lateral y eventos.** Es la mejora más directamente compatible con lo que Picchio ya tiene: predicción de expertos, caché host, caché GPU y slabs reutilizables. Ataca el coste de subir misses sin exigir que una CPU limitada sustituya a la GPU.

**Mantener P2 como experimento condicionado.** La idea híbrida de tandem es potente, pero en nuestro equipo hay que demostrar primero que el CPU puede procesar misses host-resident más rápido de lo que cuesta subirlos y que no provoca más swap. Los resultados de tandem en RTX 4090/64 GB no se pueden extrapolar a la RTX 3050/14 GiB.

**No portar doorbell ni MTP ahora.** Su complejidad y consumo de memoria no se justifican antes de tener una ruta de prefetch y una medición limpia del split.

No se propone aumentar el límite pinned de 128 MiB como parte de esta fase. El presupuesto debe cubrir buffers de transferencia realmente en vuelo y permanecer separado de los presupuestos de RAM pageable y VRAM.

## 11. Licencia

El README de tandem.cpp declara MIT y especifica que el parche modifica llama.cpp, también MIT. Aun así, cualquier copia literal debe conservar avisos y atribución de los autores originales y respetar los avisos de los archivos base. La propuesta para Picchio es adoptar los mecanismos y contratos, no incorporar el parche de llama.cpp.

---

**Resultado esperado de esta investigación:** un plan acotado y medible. La primera implementación debe intentar ocultar latencia de transferencia GPU; solo después decidir si el reparto CPU/GPU aporta algo en el hardware real.
