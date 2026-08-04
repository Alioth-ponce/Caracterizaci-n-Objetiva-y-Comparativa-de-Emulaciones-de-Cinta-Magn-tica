[README.md](https://github.com/user-attachments/files/30723759/README.md)
# Análisis Científico Iron Deck — Caracterización Objetiva de Emulaciones de Cinta Magnética

> **Artículo publicado:** *Caracterización Objetiva y Comparativa de Emulaciones de Cinta Magnética: Protocolos de Calibración, Mantenimiento de Respuesta frente a Ganancia y Compresión No Lineal por Frecuencia*  
> **Autor:** Alioth Ponce Aranda — APA Audio Labs  
> **Contacto:** aliothponce@gmail.com

---

## Descripción

Este repositorio contiene el pipeline completo de análisis DSP para la evaluación objetiva y comparativa de dos emulaciones de cinta magnética distribuidas de forma gratuita:

- **Iron Deck** (M Media Audio) — emulación paramétrica de dos canales con estándares NAB e IEC.
- **Dispositivo de referencia** — procesador de una firma establecida internacionalmente, con estándares NAB y CCIR.

El estudio examina:

| Análisis | Métricas |
|---|---|
| Retrato tonal nominal | Función de transferencia H(f) por estándar EQ y velocidad |
| Curva de transferencia estática | Input RMS vs Output RMS (dBFS) a 100 Hz y 1 kHz |
| Compresión no lineal por frecuencia | Δ_C = Output − Input a 100 Hz, 1 kHz y 5 kHz |
| Distorsión armónica total (THD) | THD (%) vs nivel de entrada a 1 kHz |
| Prueba de aliasing | Espectro de 5 kHz a saturación máxima |
| Distribución espectral | Energía por bandas de octava, ruido rosa |
| Estabilidad digital | Deriva de frecuencia vs tono patrón (NAB 3000 Hz / DIN-IEC 3150 Hz) |
| Mantenimiento de calibración | ΔH(f) bajo excursión de ganancia (drive sweep) |

---

## Estructura del Repositorio

```
Analsis_Iron_Desk/
│
├── analisis_cientifico.py        # Pipeline principal de análisis DSP
├── generar_senales.py            # Generación de señales de prueba
├── Documentacion.tex             # Artículo científico en formato IEEE (LaTeX) [ES]
├── Documentacion.pdf             # PDF compilado del artículo [ES]
├── Documentation_EN.tex          # Artículo científico en formato IEEE (LaTeX) [EN]
├── Documentation_EN.pdf          # PDF compilado del artículo [EN]
├── README.md
├── .gitignore
│
├── Comparativa/                  # Proyecto REAPER con los renders de cada plugin
│   ├── Comparativa.rpp           # Sesión REAPER
│   ├── Backups/
│   └── Media/                    # Archivos WAV procesados por cada plugin (~200 MB c/u)
│       ├── Iron Deck-*.wav
│       └── Dispositivo de referencia-*.wav
│
├── senales_prueba/               # Señales de referencia y archivo índice
│   ├── senales_prueba_completas.wav
│   └── indice_senales.txt
│
└── resultados_analisis/          # Gráficas PNG generadas automáticamente
    ├── 01a_Retrato_Tonal_IronDeck.png
    ├── 01b_Retrato_Tonal_Referencia.png
    ├── 02_Efecto_Velocidad_Cinta_IronDeck.png
    ├── 03_Efecto_Curva_EQ_IronDeck.png
    ├── 05_THD_vs_Nivel.png
    ├── 06_Aliasing_5kHz_Saturacion_Extrema.png
    ├── 07_Distribucion_Espectral_RuidoRosa.png
    └── 09_Mantenimiento_Calibracion_vs_Ganancia.png
```

> **Nota:** Los archivos WAV de audio (~200 MB cada uno) están excluidos del seguimiento de Git por su tamaño. Se pueden regenerar con `generar_senales.py` y procesando las señales a través de cada plugin en `Comparativa/Comparativa.rpp`.

---

## Convención de Nombres de Archivos WAV

```
{Dispositivo}-{IPS}_IPS-EQ_{curva}-{P1}_{V1}-{P2}_{V2}.wav
```

**Ejemplos:**
```
Iron Deck-15_IPS-EQ_NAB-REP_0-REC_5.wav
Iron Deck-30_IPS-EQ_IEC-REP_-12-REC_10.wav
Dispositivo de referencia-15_IPS-EQ_NAB-Input_0-Output_0.wav
Dispositivo de referencia-15_IPS-EQ_CCIR-Input_-12-Output_12.wav
```

---

## Requisitos

```
Python >= 3.9
numpy
scipy
matplotlib
```

Instalación:

```bash
pip install numpy scipy matplotlib
```

---

## Uso

### 1. Generar señales de prueba

```bash
python generar_senales.py
```

Genera `senales_prueba/senales_prueba_completas.wav` e `indice_senales.txt`.

### 2. Procesar señales en los plugins

Renderizar las señales de prueba a través de cada plugin en REAPER (`Comparativa/Comparativa.rpp`) con las combinaciones de parámetros deseadas, respetando la convención de nombres. Los renders van a `Comparativa/Media/`.

### 3. Ejecutar el análisis

```bash
python analisis_cientifico.py
```

O especificando un directorio diferente:

```bash
python analisis_cientifico.py /ruta/a/los/archivos
```

Los resultados (gráficas PNG) se guardan en `resultados_analisis/`.

### 4. Compilar los artículos (LaTeX)

Para la versión en español:
```bash
pdflatex Documentacion.tex
pdflatex Documentacion.tex   # segunda pasada para referencias
```

Para la versión en inglés:
```bash
pdflatex Documentation_EN.tex
pdflatex Documentation_EN.tex   # segunda pasada para referencias
```

---

## Resultados

Los resultados incluyen **7 gráficas** en formato PNG (300 dpi) listas para publicación. Los datos numéricos de las tablas están integrados directamente en `Documentacion.tex` (transferencia estática, compresión por frecuencia y estabilidad digital).

---

## Licencia

Este trabajo es de uso académico y de investigación. Si utiliza este material, por favor cite el artículo correspondiente.

---

## Contacto

**Alioth Ponce Aranda** — APA Audio Labs  
aliothponce@gmail.com
