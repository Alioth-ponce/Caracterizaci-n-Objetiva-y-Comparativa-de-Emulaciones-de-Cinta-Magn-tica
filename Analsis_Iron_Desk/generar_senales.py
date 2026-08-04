"""
Generacion de senales de prueba para la caracterizacion objetiva de Iron Deck.

Metodologia descrita en:
  "Caracterizacion objetiva de una emulacion de cinta magnetica gratuita:
   metodologia de medicion en el dominio del tiempo y la frecuencia
   aplicada a Iron Deck"

Senales generadas (Seccion 4.1):
  1. Barrido senoidal logaritmico (sine sweep) de 1 Hz a fs/2
  2. Tonos puros a 100 Hz, 1 kHz y 5 kHz
  3. Senal de impulso
  4. Ruido rosa
  5. Senal multitono
  6. Tonos de referencia a 3 000 Hz y 3 150 Hz (calibracion NAB / DIN-IEC)

Barrido de niveles (Seccion 4.2):
  Todas las senales se generan a los 5 niveles de entrada:
    -30, -18, -12, -6 y 0 dBFS
  para permitir la caracterizacion completa de:
    - Curva de ganancia estatica (Seccion 4.4)
    - Respuesta en frecuencia a distintos niveles de drive
    - THD en funcion del nivel
    - Distribucion de energia por bandas segun nivel

Formato de salida: WAV 32-bit float.
"""

import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp


# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48_000          # Hz  -- flujo de mastering tipico
DURACION_SWEEP = 30.0         # segundos
DURACION_TONO = 5.0           # segundos
DURACION_IMPULSO = 1.0        # segundos
DURACION_RUIDO = 30.0         # segundos
DURACION_MULTITONO = 10.0     # segundos
FADE_MS = 10.0                # milisegundos -- fundido de entrada/salida

# Frecuencias de tonos puros (Hz)
FRECUENCIAS_TONOS = [100, 1_000, 5_000]

# Niveles de entrada (dBFS) -- Seccion 4.1 / 4.2
NIVELES_DBFS = [-30, -18, -12, -6, 0]

# Tonos de referencia para estabilidad de frecuencia (Hz)
TONOS_REFERENCIA = [3_000, 3_150]

# Frecuencias para la senal multitono (una por octava, 31.25 Hz a 16 kHz)
FRECUENCIAS_MULTITONO = [
    31.25, 62.5, 125, 250, 500, 1_000,
    2_000, 4_000, 8_000, 16_000,
]

# Marcador / piloto
DURACION_PILOTO = 0.25        # segundos
FRECUENCIA_PILOTO = 1_000.0   # Hz
SILENCIO_ENTRE_S = 2.0        # segundos de silencio entre senales
SILENCIO_POST_PILOTO = 0.5    # segundos tras el tono piloto


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def db_a_lineal(db: float) -> float:
    """Convierte dBFS a amplitud lineal."""
    return 10.0 ** (db / 20.0)


def aplicar_fade(senal: np.ndarray, fs: int, fade_ms: float = FADE_MS) -> np.ndarray:
    """Aplica un fundido de entrada y salida (raised-cosine) para evitar clics."""
    n_fade = int(fs * fade_ms / 1_000.0)
    if n_fade == 0 or len(senal) < 2 * n_fade:
        return senal
    ventana = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
    senal = senal.copy()
    senal[:n_fade] *= ventana
    senal[-n_fade:] *= ventana[::-1]
    return senal


def escalar_a_nivel(senal: np.ndarray, nivel_dbfs: float) -> np.ndarray:
    """Escala una senal normalizada (pico = 1.0) al nivel dBFS indicado."""
    pico = np.max(np.abs(senal))
    if pico > 0:
        senal = senal * (db_a_lineal(nivel_dbfs) / pico)
    return senal


def guardar_wav(ruta: str, senal: np.ndarray, fs: int) -> None:
    """Guarda una senal como archivo WAV 32-bit float estereo."""
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    senal_estereo = np.column_stack((senal, senal))
    wavfile.write(ruta, fs, senal_estereo.astype(np.float32))
    print(f"  [OK] {ruta}  ({len(senal)/fs:.2f} s, {fs} Hz, float32 estereo)")


def formatear_tiempo(segundos: float) -> str:
    """Convierte segundos a formato MM:SS.mmm"""
    m = int(segundos) // 60
    s = segundos - m * 60
    return f"{m:02d}:{s:06.3f}"


def generar_piloto(fs: int) -> np.ndarray:
    """Tono piloto corto (1 kHz, -12 dBFS, 250 ms) como marcador audible."""
    t = np.arange(int(fs * DURACION_PILOTO)) / fs
    senal = db_a_lineal(-12.0) * np.sin(2.0 * np.pi * FRECUENCIA_PILOTO * t)
    return aplicar_fade(senal, fs)


def generar_silencio(fs: int, duracion: float = SILENCIO_ENTRE_S) -> np.ndarray:
    """Genera un buffer de silencio de la duracion indicada."""
    return np.zeros(int(fs * duracion))


# ---------------------------------------------------------------------------
# 1. Barrido senoidal logaritmico (sine sweep)
# ---------------------------------------------------------------------------

def generar_sweep(fs: int, duracion: float, nivel_dbfs: float = 0.0,
                  f_inicio: float = 1.0,
                  f_fin: float | None = None) -> np.ndarray:
    """
    Genera un barrido senoidal logaritmico de f_inicio a f_fin.

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion en segundos.
    nivel_dbfs : float
        Nivel pico en dBFS.
    f_inicio : float
        Frecuencia inicial (Hz). Por defecto 1 Hz para cubrir subgraves.
    f_fin : float | None
        Frecuencia final (Hz). Por defecto fs/2 (Nyquist).

    Retorna
    -------
    np.ndarray  (float64)
    """
    if f_fin is None:
        f_fin = fs / 2.0
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    senal = chirp(t, f0=f_inicio, f1=f_fin, t1=duracion, method="logarithmic")
    senal = escalar_a_nivel(senal, nivel_dbfs)
    return aplicar_fade(senal, fs)


# ---------------------------------------------------------------------------
# 2. Tonos puros
# ---------------------------------------------------------------------------

def generar_tono(fs: int, duracion: float, frecuencia: float,
                 nivel_dbfs: float) -> np.ndarray:
    """
    Genera un tono senoidal puro a la frecuencia y nivel indicados.

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion en segundos.
    frecuencia : float
        Frecuencia del tono (Hz).
    nivel_dbfs : float
        Nivel pico en dBFS.

    Retorna
    -------
    np.ndarray  (float64)
    """
    t = np.arange(int(fs * duracion)) / fs
    amplitud = db_a_lineal(nivel_dbfs)
    senal = amplitud * np.sin(2.0 * np.pi * frecuencia * t)
    return aplicar_fade(senal, fs)


# ---------------------------------------------------------------------------
# 3. Senal de impulso
# ---------------------------------------------------------------------------

def generar_impulso(fs: int, duracion: float,
                    nivel_dbfs: float = 0.0) -> np.ndarray:
    """
    Genera un impulso unitario (delta de Dirac discreto) centrado
    en la mitad del buffer, escalado al nivel indicado.

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion total del archivo en segundos.
    nivel_dbfs : float
        Nivel pico del impulso en dBFS.

    Retorna
    -------
    np.ndarray  (float64)
    """
    n = int(fs * duracion)
    senal = np.zeros(n)
    senal[n // 2] = db_a_lineal(nivel_dbfs)
    return senal


# ---------------------------------------------------------------------------
# 4. Ruido rosa
# ---------------------------------------------------------------------------

def generar_ruido_rosa(fs: int, duracion: float,
                       nivel_dbfs: float = -6.0) -> np.ndarray:
    """
    Genera ruido rosa (pendiente espectral de -3 dB/oct) mediante filtrado
    en el dominio frecuencial de ruido blanco gaussiano.

    Se normaliza al nivel pico indicado.

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion en segundos.
    nivel_dbfs : float
        Nivel pico en dBFS.

    Retorna
    -------
    np.ndarray  (float64)
    """
    n = int(fs * duracion)
    # Generar ruido blanco gaussiano
    blanco = np.random.default_rng(seed=42).standard_normal(n)

    # Transformar al dominio frecuencial
    X = np.fft.rfft(blanco)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Filtro 1/sqrt(f) -> pendiente de -3 dB/oct
    with np.errstate(divide="ignore", invalid="ignore"):
        filtro = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 0.0)
    X *= filtro

    senal = np.fft.irfft(X, n=n)
    # Normalizar al nivel pico indicado
    senal = escalar_a_nivel(senal, nivel_dbfs)
    return aplicar_fade(senal, fs)


# ---------------------------------------------------------------------------
# 5. Senal multitono
# ---------------------------------------------------------------------------

def generar_multitono(fs: int, duracion: float,
                      frecuencias: list[float] | None = None,
                      nivel_dbfs: float = -12.0) -> np.ndarray:
    """
    Genera una senal multitono: superposicion de senoidales a las
    frecuencias indicadas, con fases aleatorias para minimizar el
    factor de cresta (crest factor).

    Se normaliza al nivel pico indicado.

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion en segundos.
    frecuencias : list[float] | None
        Lista de frecuencias (Hz). Por defecto, una por octava de
        31.25 Hz a 16 kHz.
    nivel_dbfs : float
        Nivel pico en dBFS.

    Retorna
    -------
    np.ndarray  (float64)
    """
    if frecuencias is None:
        frecuencias = FRECUENCIAS_MULTITONO

    t = np.arange(int(fs * duracion)) / fs
    rng = np.random.default_rng(seed=123)

    senal = np.zeros_like(t)
    for f in frecuencias:
        fase = rng.uniform(0, 2.0 * np.pi)
        senal += np.sin(2.0 * np.pi * f * t + fase)

    # Normalizar al nivel pico indicado
    senal = escalar_a_nivel(senal, nivel_dbfs)
    return aplicar_fade(senal, fs)


# ---------------------------------------------------------------------------
# 6. Tonos de referencia (3 000 Hz y 3 150 Hz)
# ---------------------------------------------------------------------------

def generar_tono_referencia(fs: int, duracion: float,
                            frecuencia: float,
                            nivel_dbfs: float = -6.0) -> np.ndarray:
    """
    Genera un tono de referencia para la prueba de estabilidad de
    frecuencia (calibracion NAB / DIN-IEC).

    Parametros
    ----------
    fs : int
        Tasa de muestreo.
    duracion : float
        Duracion en segundos.
    frecuencia : float
        Frecuencia del tono de referencia (Hz).
    nivel_dbfs : float
        Nivel pico en dBFS.

    Retorna
    -------
    np.ndarray  (float64)
    """
    return generar_tono(fs, duracion, frecuencia, nivel_dbfs=nivel_dbfs)


# ---------------------------------------------------------------------------
# Construir lista completa de senales etiquetadas
# ---------------------------------------------------------------------------

def construir_senales(fs: int) -> list[tuple[str, np.ndarray]]:
    """
    Genera TODAS las senales de prueba descritas en las Secciones 4.1 y 4.2
    del documento, a todos los niveles de entrada requeridos.

    Estructura del archivo combinado:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SECCION A - BARRIDOS SENOIDALES (respuesta en frecuencia y fase)
      Sweep a -30, -18, -12, -6, 0 dBFS

    SECCION B - TONOS PUROS (THD + curva de ganancia estatica)
      100 Hz  a -30, -18, -12, -6, 0 dBFS
      1 kHz   a -30, -18, -12, -6, 0 dBFS
      5 kHz   a -30, -18, -12, -6, 0 dBFS

    SECCION C - IMPULSOS (respuesta al transitorio)
      Impulso a -30, -18, -12, -6, 0 dBFS

    SECCION D - RUIDO ROSA (distribucion de energia por bandas)
      Ruido rosa a -30, -18, -12, -6, 0 dBFS

    SECCION E - MULTITONO (comportamiento multifrequencia)
      Multitono a -30, -18, -12, -6, 0 dBFS

    SECCION F - TONOS DE REFERENCIA (estabilidad de frecuencia)
      3 000 Hz a -6 dBFS (NAB)
      3 150 Hz a -6 dBFS (DIN/IEC)
    """
    senales: list[tuple[str, np.ndarray]] = []

    # ===== SECCION A: Barridos senoidales a 5 niveles =====
    for nivel in NIVELES_DBFS:
        senales.append((
            f"[A] Sweep log  1 Hz - {fs // 2} Hz  "
            f"{nivel:+d} dBFS  ({DURACION_SWEEP:.0f} s)",
            generar_sweep(fs, DURACION_SWEEP, nivel_dbfs=nivel),
        ))

    # ===== SECCION B: Tonos puros — 3 frecuencias x 5 niveles =====
    for freq in FRECUENCIAS_TONOS:
        for nivel in NIVELES_DBFS:
            senales.append((
                f"[B] Tono {freq} Hz  {nivel:+d} dBFS  "
                f"({DURACION_TONO:.0f} s)",
                generar_tono(fs, DURACION_TONO, freq, nivel),
            ))

    # ===== SECCION C: Impulsos a 5 niveles =====
    for nivel in NIVELES_DBFS:
        senales.append((
            f"[C] Impulso  {nivel:+d} dBFS  ({DURACION_IMPULSO:.0f} s)",
            generar_impulso(fs, DURACION_IMPULSO, nivel_dbfs=nivel),
        ))

    # ===== SECCION D: Ruido rosa a 5 niveles =====
    for nivel in NIVELES_DBFS:
        senales.append((
            f"[D] Ruido rosa  {nivel:+d} dBFS pico  "
            f"({DURACION_RUIDO:.0f} s)",
            generar_ruido_rosa(fs, DURACION_RUIDO, nivel_dbfs=nivel),
        ))

    # ===== SECCION E: Multitono a 5 niveles =====
    for nivel in NIVELES_DBFS:
        senales.append((
            f"[E] Multitono  10 frec  {nivel:+d} dBFS pico  "
            f"({DURACION_MULTITONO:.0f} s)",
            generar_multitono(fs, DURACION_MULTITONO, nivel_dbfs=nivel),
        ))

    # ===== SECCION F: Tonos de referencia a -6 dBFS =====
    # Nivel fijo: la prueba de estabilidad de frecuencia no depende del
    # nivel sino de la precision frecuencial del dispositivo.
    for freq in TONOS_REFERENCIA:
        senales.append((
            f"[F] Referencia {freq} Hz  -6 dBFS  ({DURACION_TONO:.0f} s)",
            generar_tono_referencia(fs, DURACION_TONO, freq, nivel_dbfs=-6.0),
        ))

    return senales


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def generar_todas(directorio_salida: str, fs: int = SAMPLE_RATE,
                  individuales: bool = False) -> None:
    """
    Genera el conjunto completo de senales de prueba.

    Produce un archivo WAV combinado con todas las pruebas concatenadas
    (con tono piloto + silencio entre cada una) y un archivo de indice .txt.

    Si `individuales=True`, tambien guarda cada senal como WAV independiente.
    """
    print(f"\n{'='*70}")
    print(f"  Generacion de senales de prueba - Iron Deck")
    print(f"  Tasa de muestreo : {fs} Hz")
    print(f"  Profundidad      : 32-bit float")
    print(f"  Directorio       : {directorio_salida}")
    print(f"{'='*70}\n")

    # Construir todas las senales
    print("[*] Generando senales...")
    senales = construir_senales(fs)
    piloto = generar_piloto(fs)
    silencio = generar_silencio(fs)
    pausa = generar_silencio(fs, SILENCIO_POST_PILOTO)

    # --- Archivo combinado ---
    print(f"\n[*] Construyendo archivo combinado ({len(senales)} senales)...\n")

    seccion_actual = ""
    fragmentos: list[np.ndarray] = []
    indice: list[tuple[float, float, str]] = []
    cursor = 0  # posicion actual en muestras

    for i, (etiqueta, senal) in enumerate(senales):
        # Detectar cambio de seccion para imprimir encabezado
        sec = etiqueta[1]  # letra A, B, C, D, E, F
        if sec != seccion_actual:
            seccion_actual = sec
            nombres_seccion = {
                "A": "BARRIDOS SENOIDALES (respuesta en frecuencia)",
                "B": "TONOS PUROS (THD + ganancia estatica)",
                "C": "IMPULSOS (respuesta al transitorio)",
                "D": "RUIDO ROSA (distribucion de energia)",
                "E": "MULTITONO (comportamiento multifrecuencia)",
                "F": "TONOS DE REFERENCIA (estabilidad de frecuencia)",
            }
            print(f"  --- SECCION {sec}: {nombres_seccion.get(sec, '')} ---")

        # Silencio entre senales (excepto antes de la primera)
        if i > 0:
            fragmentos.append(silencio)
            cursor += len(silencio)

        # Tono piloto como marcador
        fragmentos.append(piloto)
        cursor += len(piloto)

        # Pausa post-piloto
        fragmentos.append(pausa)
        cursor += len(pausa)

        # Senal de prueba
        inicio = cursor / fs
        fragmentos.append(senal)
        cursor += len(senal)
        fin = cursor / fs

        indice.append((inicio, fin, etiqueta))
        print(f"  {i+1:3d}/{len(senales)}  "
              f"[{formatear_tiempo(inicio)} - {formatear_tiempo(fin)}]  "
              f"{etiqueta}")

    # Concatenar todo
    combinado = np.concatenate(fragmentos)
    duracion_total = len(combinado) / fs

    # Guardar WAV combinado
    ruta_wav = os.path.join(directorio_salida, "senales_prueba_completas.wav")
    guardar_wav(ruta_wav, combinado, fs)

    # --- Guardar indice de tiempos ---
    ruta_indice = os.path.join(directorio_salida, "indice_senales.txt")
    os.makedirs(os.path.dirname(ruta_indice) or ".", exist_ok=True)

    with open(ruta_indice, "w", encoding="utf-8") as f:
        f.write("INDICE DE SENALES DE PRUEBA - IRON DECK\n")
        f.write(f"Tasa de muestreo: {fs} Hz | 32-bit float\n")
        f.write(f"Duracion total:   {formatear_tiempo(duracion_total)}\n")
        f.write(f"Total de senales: {len(senales)}\n")
        f.write("=" * 90 + "\n\n")

        # Resumen de secciones
        f.write("ESTRUCTURA DEL ARCHIVO\n")
        f.write("-" * 90 + "\n")
        f.write("Seccion A: Barridos senoidales log (1 Hz - Nyquist) x 5 niveles\n")
        f.write("           Para: respuesta en frecuencia y fase (deconvolucion)\n")
        f.write("Seccion B: Tonos puros (100 Hz, 1 kHz, 5 kHz) x 5 niveles\n")
        f.write("           Para: THD, curva de ganancia estatica\n")
        f.write("Seccion C: Impulsos x 5 niveles\n")
        f.write("           Para: respuesta al transitorio\n")
        f.write("Seccion D: Ruido rosa x 5 niveles\n")
        f.write("           Para: distribucion de energia por bandas (Welch PSD)\n")
        f.write("Seccion E: Multitono (10 frecuencias) x 5 niveles\n")
        f.write("           Para: comportamiento multifrecuencia simultaneo\n")
        f.write("Seccion F: Tonos de referencia (3000 Hz NAB, 3150 Hz DIN/IEC)\n")
        f.write("           Para: prueba de estabilidad de frecuencia\n")
        f.write("-" * 90 + "\n\n")

        # Niveles evaluados
        niveles_str = ", ".join(f"{n:+d}" for n in NIVELES_DBFS)
        f.write(f"Niveles de entrada evaluados: {niveles_str} dBFS\n\n")

        # Tabla de tiempos
        f.write(f"{'#':>3}  {'INICIO':>10}  {'FIN':>10}  SENAL\n")
        f.write("=" * 90 + "\n")

        sec_actual = ""
        for i, (ini, fin, etiq) in enumerate(indice):
            sec = etiq[1]
            if sec != sec_actual:
                sec_actual = sec
                if i > 0:
                    f.write("-" * 90 + "\n")
            f.write(f"{i+1:3d}  {formatear_tiempo(ini):>10}  "
                    f"{formatear_tiempo(fin):>10}  {etiq}\n")
        f.write("=" * 90 + "\n")

        f.write(f"\nNota: cada senal esta precedida por un tono piloto de "
                f"1 kHz (250 ms, -12 dBFS)\n"
                f"      seguido de {SILENCIO_POST_PILOTO*1000:.0f} ms de "
                f"silencio. Entre senales hay "
                f"{SILENCIO_ENTRE_S:.0f} s de silencio.\n")

        # Instrucciones de uso para la DAW
        f.write("\n" + "=" * 90 + "\n")
        f.write("INSTRUCCIONES DE USO EN DAW (Seccion 4.2 / 4.3)\n")
        f.write("-" * 90 + "\n")
        f.write("1. Importar este archivo en la DAW como pista de referencia.\n")
        f.write("2. Dispositivo de referencia: procesar con input en:\n")
        f.write("     -12, -3, +6, +15, +24 dB  (5 puntos del rango -12..+24)\n")
        f.write("3. Iron Deck: procesar con las siguientes combinaciones:\n")
        f.write("   a) REC (drive 0-10) en 5 puntos: 0, 2.5, 5, 7.5, 10\n")
        f.write("   b) REP (salida -12..+12 dB) en 5 puntos: -12, -6, 0, +6, +12\n")
        f.write("   c) Modos de recorrido: REPRO, INPUT, SYNC  (cada uno por "
                "separado)\n")
        f.write("4. Exportar cada resultado manteniendo fs y bit-depth.\n")
        f.write("=" * 90 + "\n")

    print(f"\n  [OK] Indice: {ruta_indice}")

    # --- Archivos individuales (opcional) ---
    if individuales:
        print("\n[*] Guardando archivos individuales...")
        for i, (etiqueta, senal) in enumerate(senales):
            # Extraer seccion y construir nombre seguro
            sec = etiqueta[1]  # A, B, C, D, E, F
            carpetas = {
                "A": "A_sweeps",
                "B": "B_tonos",
                "C": "C_impulsos",
                "D": "D_ruido_rosa",
                "E": "E_multitono",
                "F": "F_referencia",
            }
            carpeta = carpetas.get(sec, "otros")
            nombre = (etiqueta[4:]  # quitar "[X] "
                      .replace(" ", "_").replace("/", "-")
                      .replace("+", "p").replace("(", "").replace(")", "")
                      .replace(",", "")
                      + ".wav")
            guardar_wav(os.path.join(directorio_salida, carpeta, nombre),
                        senal, fs)

    # --- Resumen ---
    n_sweeps = len(NIVELES_DBFS)
    n_tonos = len(FRECUENCIAS_TONOS) * len(NIVELES_DBFS)
    n_impulsos = len(NIVELES_DBFS)
    n_rosa = len(NIVELES_DBFS)
    n_multi = len(NIVELES_DBFS)
    n_ref = len(TONOS_REFERENCIA)
    total = len(senales)

    print(f"\n{'='*70}")
    print(f"  Generacion completada")
    print(f"  Archivo combinado : {ruta_wav}")
    print(f"  Duracion total    : {formatear_tiempo(duracion_total)}")
    print(f"  Senales incluidas : {total}")
    print(f"    [A]  {n_sweeps:2d} barridos senoidales  "
          f"(1 sweep x {len(NIVELES_DBFS)} niveles)")
    print(f"    [B]  {n_tonos:2d} tonos puros          "
          f"({len(FRECUENCIAS_TONOS)} frec x {len(NIVELES_DBFS)} niveles)")
    print(f"    [C]  {n_impulsos:2d} impulsos             "
          f"(1 impulso x {len(NIVELES_DBFS)} niveles)")
    print(f"    [D]  {n_rosa:2d} ruido rosa           "
          f"(1 ruido x {len(NIVELES_DBFS)} niveles)")
    print(f"    [E]  {n_multi:2d} multitono            "
          f"(1 multi x {len(NIVELES_DBFS)} niveles)")
    print(f"    [F]  {n_ref:2d} tonos de referencia  "
          f"(3000 Hz NAB + 3150 Hz DIN/IEC)")
    if individuales:
        print(f"  Archivos indiv.   : {total} WAV en subcarpetas")
    print(f"  Indice de tiempos : {ruta_indice}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera senales de prueba para la caracterizacion "
                    "objetiva de Iron Deck.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python generar_senales.py
  python generar_senales.py --fs 96000 --salida ./mis_senales
  python generar_senales.py --individuales
        """,
    )
    parser.add_argument(
        "--fs", type=int, default=SAMPLE_RATE,
        help=f"Tasa de muestreo en Hz (por defecto: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--salida", type=str, default="senales_prueba",
        help="Directorio de salida (por defecto: senales_prueba)",
    )
    parser.add_argument(
        "--individuales", action="store_true",
        help="Tambien guardar cada senal como archivo WAV independiente",
    )
    args = parser.parse_args()
    generar_todas(args.salida, args.fs, args.individuales)


if __name__ == "__main__":
    main()
