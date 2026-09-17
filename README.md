# llama.cpp en Windows: instalación, servidor web y API local

Guía práctica para ejecutar modelos en formato GGUF con `llama.cpp` en Windows, aprovechar la aceleración CUDA cuando esté disponible y exponer el modelo mediante una interfaz web y una API local compatible con el formato de OpenAI.

## Índice

- [Requisitos previos](#requisitos-previos)
- [1. Descargar llama.cpp](#1-descargar-llamacpp)
- [2. Preparar las carpetas](#2-preparar-las-carpetas)
- [3. Descargar un modelo GGUF](#3-descargar-un-modelo-gguf)
- [4. Comprobar que CUDA detecta la GPU](#4-comprobar-que-cuda-detecta-la-gpu)
- [5. Configurar el archivo BAT](#5-configurar-el-archivo-bat)
- [6. Arrancar el servidor](#6-arrancar-el-servidor)
- [7. Usar la interfaz web](#7-usar-la-interfaz-web)
- [8. Usar la API compatible con OpenAI](#8-usar-la-api-compatible-con-openai)
- [9. Medir el rendimiento con llama-bench](#9-medir-el-rendimiento-con-llama-bench)
- [10. Interpretar los tokens por segundo](#10-interpretar-los-tokens-por-segundo)
- [Solución de problemas](#solución-de-problemas)
- [Actualizar llama.cpp](#actualizar-llamacpp)

## Requisitos previos

- Windows 10 u 11 de 64 bits.
- Una GPU NVIDIA compatible y sus controladores actualizados, si se va a utilizar CUDA.
- Espacio libre suficiente para los binarios, el runtime de CUDA y uno o varios modelos GGUF.
- Un navegador web.
- Opcional: `nvidia-smi`, incluido normalmente con el controlador de NVIDIA, para revisar el estado de la GPU.

Esta guía utiliza los binarios precompilados oficiales. No es necesario instalar Visual Studio ni compilar el proyecto.

## 1. Descargar llama.cpp

1. Abre la página oficial de [releases de llama.cpp](https://github.com/ggml-org/llama.cpp/releases).
2. En la release más reciente, localiza el apartado **Windows**.
3. Para utilizar una GPU NVIDIA, descarga estos **dos archivos de la misma release y de la misma variante CUDA**:
   - **Windows x64 (CUDA 12 o CUDA 13)**: contiene `llama-server.exe`, `llama-bench.exe` y el resto de ejecutables.
   - **CUDA DLLs x64** que aparece junto a esa build: es el paquete `cudart` con las DLL necesarias.
4. Crea una carpeta nueva, por ejemplo `C:\llama.cpp`.
5. Extrae el contenido de ambos ZIP dentro de esa misma carpeta. Al terminar, `llama-server.exe`, `ggml-cuda.dll`, `cudart64_*.dll`, `cublas64_*.dll` y `cublasLt64_*.dll` deben quedar disponibles en el mismo directorio de ejecución.

> [!IMPORTANT]
> Descarga siempre la variante **x64** para un PC Windows convencional con procesador Intel o AMD. No mezcles una build x64 con el paquete `cudart` **ARM64**. Tampoco combines archivos de releases, arquitecturas o versiones CUDA distintas, aunque sus nombres parezcan similares.

Si no necesitas CUDA, puedes descargar la build **Windows x64 (CPU)**. En ese caso no hace falta el paquete `cudart`, pero el rendimiento será distinto y `--list-devices` no mostrará un dispositivo CUDA.

### Elegir entre CUDA 12 y CUDA 13

Empieza por una de las combinaciones x64 publicadas en la release actual. Si el backend no detecta la GPU, prueba la otra variante, siempre sustituyendo **tanto los binarios como sus DLL** y usando una carpeta limpia.

No copies DLL antiguas sobre una instalación nueva. Mantener cada combinación en su propia carpeta evita dependencias mezcladas y facilita volver a una versión que ya funcionaba.

## 2. Preparar las carpetas

Una estructura sencilla puede ser:

```text
C:\llama.cpp\
├── llama-server.exe
├── llama-bench.exe
├── ggml-cuda.dll
├── cudart64_*.dll
├── cublas64_*.dll
├── cublasLt64_*.dll
├── ...otros archivos de la release
└── models\
    └── model.gguf
```

Los nombres exactos de las DLL cambian según la versión CUDA. Lo importante es que procedan del paquete `cudart` enlazado junto a la build elegida.

## 3. Descargar un modelo GGUF

`llama.cpp` utiliza modelos en formato **GGUF**. Pueden encontrarse, por ejemplo, en [Hugging Face](https://huggingface.co/models?library=gguf).

Antes de descargar un archivo:

- Comprueba que el repositorio corresponde al modelo y variante instruct/chat que quieres usar.
- Lee la ficha del modelo y respeta su licencia.
- Confirma que el archivo termina en `.gguf`.
- Elige un tamaño y una cuantización adecuados para la memoria disponible.
- Si el modelo está dividido en varios archivos, descarga todas las partes indicadas por el repositorio.

Guarda el modelo en `C:\llama.cpp\models\model.gguf` o adapta las rutas de los ejemplos.

### Qué significa la cuantización

La cuantización reduce la precisión de los pesos para disminuir el tamaño del modelo y el uso de memoria. A cambio, puede producir una pérdida de calidad.

Como orientación general:

| Variante | Tamaño y memoria | Calidad aproximada | Uso habitual |
| --- | --- | --- | --- |
| Q2 / Q3 | Muy bajos | Mayor pérdida | Equipos con memoria muy limitada |
| Q4 | Equilibrados | Buena para muchos usos | Punto de partida recomendado |
| Q5 / Q6 | Más altos | Más cercana al original | Cuando hay memoria suficiente |
| Q8 | Altos | Pérdida pequeña | Prioridad a calidad sobre tamaño |

Las variantes con sufijos como `K_M` o `K_S` emplean esquemas diferentes. La disponibilidad y el comportamiento dependen del modelo; revisa siempre su ficha. No basta con que el archivo quepa en disco: también hay que considerar la memoria necesaria para el contexto y la caché KV.

## 4. Comprobar que CUDA detecta la GPU

Abre **Símbolo del sistema** en la carpeta de `llama.cpp` y ejecuta:

```bat
llama-server.exe --list-devices
```

También puede comprobarse con:

```bat
llama-bench.exe --list-devices
```

Una instalación CUDA correcta debe mostrar al menos un dispositivo CUDA. El texto exacto depende de la versión.

Si aparece:

```text
Available devices:
  (none)
```

no continúes ajustando el contexto, Flash Attention ni el número de capas. Primero corrige la carga del backend CUDA; de lo contrario, el servidor puede arrancar y ejecutar el modelo por CPU.

## 5. Configurar el archivo BAT

El repositorio incluye [start-server.bat](start-server.bat). Edita únicamente las variables de la parte superior:

```bat
set "LLAMA_DIR=C:\llama.cpp"
set "MODEL_PATH=C:\llama.cpp\models\model.gguf"
set "HOST=127.0.0.1"
set "PORT=8080"
set "CTX_SIZE=8192"
```

El comando principal es:

```bat
llama-server.exe ^
  --model "%MODEL_PATH%" ^
  --n-gpu-layers all ^
  --ctx-size %CTX_SIZE% ^
  --flash-attn on ^
  --host %HOST% ^
  --port %PORT%
```

Parámetros utilizados:

- `--model`: ruta del archivo GGUF.
- `--n-gpu-layers all`: intenta descargar todas las capas posibles en la GPU.
- `--ctx-size`: tamaño máximo del contexto en tokens. Un contexto mayor consume más memoria.
- `--flash-attn on`: activa Flash Attention cuando el backend y el modelo lo permiten.
- `--host 127.0.0.1`: limita el acceso al propio equipo.
- `--port 8080`: puerto de la interfaz y de la API.

Si la memoria no es suficiente, reduce primero `CTX_SIZE`. También puedes sustituir `all` por un número para descargar solo parte de las capas en la GPU.

## 6. Arrancar el servidor

Ejecuta `start-server.bat`. La primera carga puede tardar unos segundos. La consola debe mostrar:

- la detección del backend CUDA, si corresponde;
- el dispositivo utilizado;
- la carga del modelo;
- las capas descargadas en la GPU;
- la dirección en la que escucha el servidor.

Mantén la ventana abierta mientras uses el modelo. Para detener el servidor, vuelve a esa ventana y pulsa `Ctrl+C`.

## 7. Usar la interfaz web

Con el servidor listo, abre:

```text
http://127.0.0.1:8080
```

La interfaz incluida permite conversar con el modelo y cambiar parámetros de generación sin instalar otra aplicación.

Dos comprobaciones útiles desde el navegador son:

```text
http://127.0.0.1:8080/health
http://127.0.0.1:8080/v1/models
```

`/health` informa del estado del servidor y `/v1/models` devuelve los datos del modelo cargado.

## 8. Usar la API compatible con OpenAI

`llama-server` expone rutas compatibles con el formato de OpenAI. La URL base es:

```text
http://127.0.0.1:8080/v1
```

Ejemplo con `curl` desde Símbolo del sistema:

```bat
curl http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"local-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Escribe una frase breve sobre la computación local.\"}],\"temperature\":0.7}"
```

Muchas aplicaciones que aceptan un proveedor compatible con OpenAI permiten indicar esa URL base. Si solicitan una clave y el servidor no se ha configurado con autenticación, suele poder utilizarse un valor de marcador, por ejemplo `local`.

La compatibilidad no implica que todas las funciones de un servicio remoto estén implementadas. Consulta la [documentación oficial de llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) para conocer los endpoints y límites actuales.

### Acceso desde otros equipos

Esta guía usa `127.0.0.1` para que el servidor solo sea accesible desde el mismo PC. No cambies el host a `0.0.0.0` sin entender las implicaciones: expondría el servicio a la red y requeriría configurar autenticación, cortafuegos y controles de acceso adecuados.

## 9. Medir el rendimiento con llama-bench

Cierra `llama-server` antes de ejecutar el benchmark para evitar que ambos procesos compitan por la memoria.

Desde la carpeta de `llama.cpp`:

```bat
llama-bench.exe -m "C:\llama.cpp\models\model.gguf" -ngl all
```

Para comparar de forma útil:

- usa el mismo modelo y la misma cuantización;
- mantén los mismos parámetros;
- cierra aplicaciones que estén utilizando la GPU;
- deja que el benchmark complete sus repeticiones;
- anota la versión de `llama.cpp` utilizada.

La salida separa pruebas de procesado del prompt y generación. Revisa también el backend o dispositivo indicado para confirmar que la medición no se ha realizado solo por CPU.

## 10. Interpretar los tokens por segundo

No confundas estas dos métricas:

- **Prompt processing** (`pp`): velocidad a la que se procesa el texto de entrada. Suele ser bastante más alta.
- **Text generation** (`tg`): velocidad a la que el modelo genera nuevos tokens. Es la referencia relevante al hablar de fluidez de respuesta.

Un valor alto de prompt processing no significa que la generación tenga esa misma velocidad. Para comparar configuraciones, usa por separado `pp` y `tg`, con cargas equivalentes y una salida suficientemente larga para reducir el peso del tiempo de arranque.

## Solución de problemas

### `Available devices: (none)`

1. Comprueba que descargaste la build **Windows x64 CUDA**, no la build CPU ni ARM64.
2. Confirma que el paquete `cudart` también es **x64**, corresponde a la misma variante CUDA y procede de la misma release.
3. Verifica que las DLL de CUDA están junto a `llama-server.exe` y `ggml-cuda.dll`.
4. Repite la instalación en una carpeta nueva. No reutilices DLL de otra versión.
5. Ejecuta `nvidia-smi`. Si falla o no muestra la GPU, actualiza o reinstala el controlador desde NVIDIA.
6. Prueba la otra build CUDA x64 ofrecida en la misma release, con su propio paquete `cudart`.
7. Vuelve a ejecutar `llama-server.exe --list-devices` antes de probar el modelo.

### El servidor arranca, pero todo funciona por CPU

- Busca en el registro líneas que indiquen que se cargó el backend CUDA y que se descargaron capas en la GPU.
- Confirma que se utiliza `--n-gpu-layers all` o un valor mayor que cero.
- Ejecuta `--list-devices`; que el archivo se llame “CUDA” no garantiza que el backend se haya cargado.
- No uses el resultado de una build CPU para evaluar el rendimiento esperado de CUDA.

### Error al cargar una DLL

- Extrae de nuevo ambos ZIP en una carpeta limpia.
- No renombres DLL ni copies archivos encontrados en páginas de terceros.
- Comprueba que ninguna parte de la instalación sea ARM64.
- Mantén juntos los binarios y el runtime publicados como pareja en la release.
- Revisa si el antivirus puso algún archivo en cuarentena.

### Memoria insuficiente

- Reduce `CTX_SIZE` en `start-server.bat`.
- Elige una cuantización más pequeña.
- Descarga menos capas en la GPU, por ejemplo `--n-gpu-layers 20`.
- Cierra aplicaciones que consuman VRAM.

### El modelo no responde bien en modo chat

- Usa una variante instruct/chat, no un modelo base, salvo que ese sea el objetivo.
- Verifica que el GGUF incluye una plantilla de chat compatible.
- Consulta la ficha del modelo por si requiere argumentos o una plantilla específica.

### El puerto 8080 ya está en uso

Cambia `PORT` en `start-server.bat`, por ejemplo a `8081`, y abre la nueva dirección:

```text
http://127.0.0.1:8081
```

## Actualizar llama.cpp

Las releases de `llama.cpp` cambian con frecuencia. Para actualizar con seguridad:

1. Descarga una build nueva y su paquete `cudart` correspondiente.
2. Extráelos en una carpeta nueva, sin sobrescribir la instalación anterior.
3. Ejecuta `--list-devices`.
4. Prueba el servidor y `llama-bench` con el mismo modelo.
5. Conserva la carpeta anterior hasta confirmar que todo funciona.

## Referencias oficiales

- [Releases de llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
- [Documentación de llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Documentación de llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- [Modelos y formato GGUF](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)
