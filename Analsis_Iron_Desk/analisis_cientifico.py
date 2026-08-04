"""
Analisis cientifico de la caracterizacion Iron Deck vs Dispositivo de referencia.

CAMBIOS RESPECTO A LA VERSION ORIGINAL
---------------------------------------------------------------------------
1. Los archivos procesados NO se identifican por substrings fragiles
   ("15_IPS" in nombre, etc). Se parsea la metadata real del nombre de
   archivo con una expresion regular:

       {Dispositivo}-{IPS}_IPS-EQ_{curva}-{param1}_{val1}-{param2}_{val2}.wav

   Ejemplos reales:
       Dispositivo de referencia-15_IPS-EQ_CCIR-Input_-12-Output_12.wav
       Iron Deck-30_IPS-EQ_IEC-REP_-12-REC_10.wav

   Esto elimina el bug de la version anterior donde `"REC_10".split('REC_')[1]`
   devolvia "10.wav" y rompia el sort/parsing, y hace que el script escanee
   la carpeta del proyecto y encuentre CUALQUIER combinacion de archivos
   presente, en vez de asumir una rejilla fija de 5 niveles de drive que
   nunca se genero (los datos reales tienen 3 puntos de drive, no 5).

2. Las graficas se redisenaron alrededor de la rejilla de datos que
   REALMENTE existe (3 velocidades x hasta 2 curvas EQ x 3 puntos de
   drive), y cada una responde una pregunta concreta que le importa a un
   ingeniero de mastering, con leyenda/anotacion en lenguaje llano:

     01 - Retrato tonal a ajuste nominal, 15 ips (comparacion directa)
     02 - Efecto de la velocidad de cinta en Iron Deck (7.5/15/30 ips)
     03 - Efecto de la curva NAB vs IEC en Iron Deck a 30 ips
     04 - Curva de saturacion/compresion estatica (1 kHz y 100 Hz)
     05 - THD vs nivel de entrada, por punto de drive
     06 - Prueba de aliasing a saturacion extrema (tono 5 kHz)
     07 - Distribucion espectral (ruido rosa) en ajuste nominal
     08 - Estabilidad de frecuencia (tono patron 3000/3150 Hz)
"""

import os
import re
import glob
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# CONFIGURACION DE ESTILOS CIENTIFICOS (ESTILO IEEE/LAB)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'figure.autolayout': True,
})


def configurar_eje_frecuencia(ax, title, ylabel, xlim=(20, 20000)):
    """Configura un eje X logaritmico estandar (20 Hz - 20 kHz) para publicaciones."""
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f"{int(x)}" if x < 1000 else f"{int(x/1000)}k"
    ))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)


# Función nota() eliminada. Las gráficas científicas no llevan anotaciones
# informales de mastering al pie de figura.


def limites_tight(valores, pad_frac=0.08, pad_min=0.3):
    """Calcula (lo, hi) ajustado EXACTAMENTE al rango de `valores`, con un
    margen minimo de aire. Evita que el eje quede sobredimensionado
    respecto a la informacion real (nada de rangos fijos tipo -140..0 dB
    'por si acaso' si los datos ocupan solo una fraccion de eso)."""
    valores = np.asarray(valores)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return None
    lo, hi = float(valores.min()), float(valores.max())
    pad = max((hi - lo) * pad_frac, pad_min)
    return lo - pad, hi + pad


# Centros de banda de octava usados para el analisis de energia del ruido
# rosa (Seccion IV-D del documento: "segmentada en bandas de octava").
CENTROS_OCTAVA = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def calcular_bandas_octava(f, mag_db, centros=CENTROS_OCTAVA):
    """Integra la PSD (reconstruida desde mag_db = 10*log10(Pxx)) dentro de
    cada banda de octava [fc/sqrt2, fc*sqrt2] y devuelve el nivel de banda
    en dB. Esto da una lectura mucho mas clara del comportamiento espectral
    que una curva continua punto a punto, que es ruidosa y dificil de leer."""
    f = np.asarray(f)
    Pxx = 10.0 ** (np.asarray(mag_db) / 10.0)
    niveles = []
    for fc in centros:
        f_lo, f_hi = fc / (2 ** 0.5), fc * (2 ** 0.5)
        mask = (f >= f_lo) & (f <= f_hi)
        if mask.sum() < 2:
            niveles.append(np.nan)
            continue
        _integrar = getattr(np, 'trapezoid', None) or np.trapz
        energia = _integrar(Pxx[mask], f[mask])
        niveles.append(10 * np.log10(energia + 1e-15))
    return np.array(niveles)


def etiquetas_bandas(centros=CENTROS_OCTAVA):
    return [f"{int(c)}" if c < 1000 else f"{c/1000:g}k" for c in centros]


def plot_comparacion_bandas(ax_top, ax_bot, centros, niveles_a, niveles_b, label_a, label_b, titulo):
    """Dibuja un panel de barras agrupadas por banda de octava (ax_top) y,
    debajo, la diferencia banda a banda (ax_bot) con su propio eje ajustado
    -- este segundo panel es el que hace visibles diferencias sutiles que
    se pierden a simple vista al superponer curvas casi identicas."""
    x = np.arange(len(centros))
    ancho = 0.38
    ax_top.bar(x - ancho / 2, niveles_a, ancho, label=label_a, color='tab:orange')
    ax_top.bar(x + ancho / 2, niveles_b, ancho, label=label_b, color='tab:blue')
    lim = limites_tight(np.concatenate([niveles_a, niveles_b]), pad_frac=0.15, pad_min=0.5)
    if lim:
        ax_top.set_ylim(*lim)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(etiquetas_bandas(centros))
    ax_top.set_ylabel("Nivel de banda (dB)")
    ax_top.set_title(titulo, fontweight='bold')
    ax_top.legend(fontsize=8)
    ax_top.grid(True, axis='y', linestyle='--', alpha=0.5)

    delta = niveles_a - niveles_b
    colores_delta = ['tab:red' if (np.isfinite(d) and d > 0) else 'tab:green' for d in delta]
    ax_bot.bar(x, delta, color=colores_delta)
    ax_bot.axhline(0, color='k', linewidth=0.8)
    lim_d = limites_tight(delta, pad_frac=0.2, pad_min=0.15)
    if lim_d:
        ax_bot.set_ylim(*lim_d)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(etiquetas_bandas(centros))
    ax_bot.set_xlabel("Banda de octava (Hz)")
    ax_bot.set_ylabel(f"Δ dB\n({label_a[:12]} − {label_b[:12]})", fontsize=7.5)
    ax_bot.grid(True, axis='y', linestyle='--', alpha=0.5)


# ==============================================================================
# PARSER DE METADATA DE ARCHIVO (nombre real de archivo -> dict)
# ==============================================================================
PATRON_NOMBRE = re.compile(
    r'^(?P<device>.+?)-(?P<ips>[\d.]+)_IPS-EQ_(?P<eq>[A-Za-z]+)-'
    r'(?P<p1>[A-Za-z]+)_(?P<v1>-?[\d.]+)-(?P<p2>[A-Za-z]+)_(?P<v2>-?[\d.]+)$'
)


def parsear_nombre_archivo(nombre_base):
    """
    Extrae metadata estructurada del nombre de archivo real, p.ej.:
      'Iron Deck-15_IPS-EQ_NAB-REP_-12-REC_10'
      'Dispositivo de referencia-15_IPS-EQ_CCIR-Input_-12-Output_12'

    Retorna None si el nombre no matchea el patron esperado (se ignora
    con una advertencia en vez de reventar el script).
    """
    m = PATRON_NOMBRE.match(nombre_base)
    if not m:
        return None
    d = m.groupdict()
    es_referencia = 'referencia' in d['device'].lower()
    meta = {
        'device_raw': d['device'].strip(),
        'es_referencia': es_referencia,
        'ips': float(d['ips']),
        'eq': d['eq'].upper(),
    }
    v1, v2 = float(d['v1']), float(d['v2'])
    # Normalizamos el "punto de drive" a un solo campo comparable entre
    # dispositivos: para la referencia es el nivel de Input; para Iron
    # Deck es el control REC (drive de saturacion de cinta).
    if d['p1'].upper() == 'INPUT':
        meta['input_db'] = v1
        meta['output_db'] = v2
        meta['drive'] = v1
        meta['drive_label'] = f"Input {v1:+.0f} dB"
    else:
        meta['rep_db'] = v1
        meta['rec'] = v2
        meta['drive'] = v2
        meta['drive_label'] = f"REC {v2:g}"
    return meta


def descubrir_archivos(dir_proyecto):
    """Escanea dir_proyecto y devuelve {ruta: meta} para cada WAV procesado
    reconocible. Ignora la senal original combinada y cualquier archivo
    que no siga la convencion de nombres."""
    candidatos = glob.glob(os.path.join(dir_proyecto, "*.wav"))
    archivos = {}
    for ruta in candidatos:
        base = os.path.splitext(os.path.basename(ruta))[0]
        if base == "senales_prueba_completas":
            continue
        meta = parsear_nombre_archivo(base)
        if meta is None:
            print(f"[!] Nombre no reconocido, se omite: {os.path.basename(ruta)}")
            continue
        archivos[ruta] = meta
    return archivos


def localizar_referencia(dir_proyecto):
    """Busca la senal original y el indice de tiempos en dir_proyecto o en
    dir_proyecto/senales_prueba (ambas ubicaciones son validas)."""
    candidatos_dir = [dir_proyecto, os.path.join(dir_proyecto, "senales_prueba")]
    for d in candidatos_dir:
        wav = os.path.join(d, "senales_prueba_completas.wav")
        idx = os.path.join(d, "indice_senales.txt")
        if os.path.exists(wav) and os.path.exists(idx):
            return wav, idx
    return None, None


# ==============================================================================
# PARSER DE INDICE
# ==============================================================================
def parsear_indice(ruta_indice):
    """Parsea el archivo indice_senales.txt para extraer los segmentos."""
    segmentos = []
    with open(ruta_indice, 'r', encoding='utf-8') as f:
        leyendo = False
        for linea in f:
            if "INICIO" in linea and "FIN" in linea and "SENAL" in linea:
                leyendo = True
                continue
            if leyendo:
                linea = linea.strip()
                if not linea or linea.startswith("=") or linea.startswith("-") or linea.startswith("Nota:"):
                    continue
                partes = linea.split()
                if len(partes) >= 4 and partes[0].isdigit():
                    def mmss_a_segundos(cadena):
                        m, s = cadena.split(':')
                        return int(m) * 60 + float(s)
                    try:
                        inicio_s = mmss_a_segundos(partes[1])
                        fin_s = mmss_a_segundos(partes[2])
                        etiqueta = " ".join(partes[3:])
                        segmentos.append({'inicio': inicio_s, 'fin': fin_s, 'etiqueta': etiqueta})
                    except Exception as e:
                        print(f"Error parseando linea: {linea} -> {e}")
    return segmentos


def aislar_senal(array, fs, inicio_s, fin_s):
    """Extrae la senal descartando piloto (0.25s) + pausa (0.5s) segun diseno del generador."""
    inicio_real = inicio_s + 0.75
    return array[int(inicio_real * fs):int(fin_s * fs)]


# Extraen el nivel/frecuencia de la etiqueta de forma generica (funciona
# para cualquiera de los 5 niveles, no solo -12/0 dBFS como antes).
RE_NIVEL = re.compile(r'([+-]\d+)\s*dBFS')
RE_TONO_HZ = re.compile(r'Tono\s+(\d+)\s*Hz')
RE_REF_HZ = re.compile(r'Referencia\s+(\d+)\s*Hz')


# ==============================================================================
# METRICAS DSP
# ==============================================================================
def calcular_respuesta_frecuencia(senal_dry, senal_wet, fs):
    """Respuesta en frecuencia via funcion de transferencia H = Pxy / Pxx."""
    f, Pxx = signal.csd(senal_dry, senal_dry, fs, nperseg=fs, nfft=fs * 2)
    f, Pxy = signal.csd(senal_dry, senal_wet, fs, nperseg=fs, nfft=fs * 2)
    H = Pxy / (Pxx + 1e-12)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    mag_smooth = signal.savgol_filter(mag_db, window_length=151, polyorder=2)
    return f, mag_smooth


def calcular_thd(senal_wet, fs, frec_fundamental):
    """THD (%) de un tono puro, sumando energia de armonicos 2..19."""
    f, Pxx = signal.welch(senal_wet, fs, nperseg=fs, nfft=fs * 4)
    idx_fund = np.argmin(np.abs(f - frec_fundamental))
    energia_fund = Pxx[idx_fund]
    energia_armonicos = 0
    for k in range(2, 20):
        frec_arm = frec_fundamental * k
        if frec_arm > fs / 2:
            break
        idx_arm = np.argmin(np.abs(f - frec_arm))
        energia_armonicos += Pxx[idx_arm]
    return np.sqrt(energia_armonicos / (energia_fund + 1e-12)) * 100


def medir_rms_db(senal_seca, senal_mojada):
    """RMS de entrada/salida en dBFS, normalizando si el WAV vino como entero."""
    seca = senal_seca.astype(np.float64)
    mojada = senal_mojada.astype(np.float64)
    rms_in = np.sqrt(np.mean(seca ** 2))
    rms_out = np.sqrt(np.mean(mojada ** 2))
    if np.isnan(rms_in) or np.isnan(rms_out) or rms_in <= 0:
        return None
    if seca.max() > 2.0:  # PCM entero -> normalizar a full scale
        rms_in /= 2147483648.0
        rms_out /= 2147483648.0
    return 20 * np.log10(rms_in + 1e-12), 20 * np.log10(rms_out + 1e-12)


# ==============================================================================
# HELPERS DE BUSQUEDA SOBRE LA METADATA PARSEADA
# ==============================================================================
def filtrar(resultados, **cond):
    """Devuelve las claves de `resultados` cuya metadata cumple TODAS las
    condiciones dadas, p.ej. filtrar(resultados, es_referencia=False, ips=15.0, eq='NAB')."""
    claves = []
    for k, v in resultados.items():
        meta = v['meta']
        ok = True
        for campo, valor in cond.items():
            if campo not in meta or meta[campo] != valor:
                ok = False
                break
        if ok:
            claves.append(k)
    return claves


def unico(resultados, **cond):
    claves = filtrar(resultados, **cond)
    return claves[0] if claves else None


# ==============================================================================
# MAIN: PROCESAMIENTO
# ==============================================================================
def analizar_proyecto(dir_proyecto=".", dir_resultados="resultados_analisis"):
    os.makedirs(dir_resultados, exist_ok=True)

    ruta_original, ruta_indice = localizar_referencia(dir_proyecto)
    if ruta_original is None:
        print("[X] No se encontro 'senales_prueba_completas.wav' + 'indice_senales.txt' "
              f"en {dir_proyecto} ni en {dir_proyecto}/senales_prueba")
        return

    fs, audio_dry = wavfile.read(ruta_original)
    if audio_dry.ndim > 1:
        audio_dry = audio_dry[:, 0]

    segmentos = parsear_indice(ruta_indice)
    print(f"[*] Indice cargado: {len(segmentos)} segmentos.")

    archivos = descubrir_archivos(dir_proyecto)
    print(f"[*] Encontrados {len(archivos)} archivos procesados reconocidos:")
    for ruta, meta in sorted(archivos.items(), key=lambda x: x[1]['device_raw']):
        etiqueta = "Iron Deck" if not meta['es_referencia'] else "Referencia"
        print(f"      {etiqueta:10s} | {meta['ips']:5.1f} ips | EQ {meta['eq']:5s} | {meta['drive_label']}")

    if not archivos:
        print("[X] Ningun archivo coincide con el patron esperado de nombres. Abortando.")
        return

    resultados = {}

    # --- 1. PROCESAMIENTO DSP POR ARCHIVO ---
    for ruta, meta in archivos.items():
        nombre = os.path.basename(ruta)
        print(f"Procesando: {nombre} ...")

        fs_wet, audio_wet = wavfile.read(ruta)
        if audio_wet.ndim > 1:
            audio_wet = audio_wet[:, 0]

        r = {
            'meta': meta,
            'freq_resp': {},      # nivel_dBFS -> (f, mag_db)
            'thd': {},             # frecuencia_Hz -> [(nivel, thd_pct), ...]
            'compresion': {},      # frecuencia_Hz -> [(db_in, db_out), ...]
            'psd': {},             # nivel_dBFS -> (f, mag_db)
            'estabilidad': {},     # frecuencia_patron -> frecuencia_detectada
        }
        resultados[nombre] = r

        for seg in segmentos:
            senal_seca = aislar_senal(audio_dry, fs, seg['inicio'], seg['fin'])
            senal_mojada = aislar_senal(audio_wet, fs_wet, seg['inicio'], seg['fin'])
            min_len = min(len(senal_seca), len(senal_mojada))
            if min_len <= 0:
                continue
            senal_seca = senal_seca[:min_len]
            senal_mojada = senal_mojada[:min_len]
            etiq = seg['etiqueta']

            # [A] Respuesta en frecuencia (los 5 niveles de sweep)
            if etiq.startswith("[A] Sweep"):
                m = RE_NIVEL.search(etiq)
                if m:
                    nivel = int(m.group(1))
                    f, mag = calcular_respuesta_frecuencia(senal_seca, senal_mojada, fs)
                    r['freq_resp'][nivel] = (f, mag)

            # [B] Tonos puros -> THD + curva de compresion (todas las frecuencias)
            elif etiq.startswith("[B] Tono"):
                m_hz = RE_TONO_HZ.search(etiq)
                m_niv = RE_NIVEL.search(etiq)
                if m_hz and m_niv:
                    frec = int(m_hz.group(1))
                    nivel = int(m_niv.group(1))
                    thd = calcular_thd(senal_mojada, fs, frec)
                    r['thd'].setdefault(frec, []).append((nivel, thd))
                    medida = medir_rms_db(senal_seca, senal_mojada)
                    if medida is not None:
                        r['compresion'].setdefault(frec, []).append(medida)

            # [D] Ruido rosa -> PSD por nivel
            elif etiq.startswith("[D] Ruido rosa"):
                m_niv = RE_NIVEL.search(etiq)
                if m_niv:
                    nivel = int(m_niv.group(1))
                    f, Pxx = signal.welch(senal_mojada, fs, nperseg=fs, nfft=fs * 2)
                    r['psd'][nivel] = (f, 10 * np.log10(Pxx + 1e-12))

            # [F] Tonos de referencia -> estabilidad de frecuencia
            elif etiq.startswith("[F] Referencia"):
                m_hz = RE_REF_HZ.search(etiq)
                if m_hz:
                    frec_patron = int(m_hz.group(1))
                    f, Pxx = signal.welch(senal_mojada, fs, nperseg=fs * 4)
                    idx_pico = np.argmax(Pxx)
                    r['estabilidad'][frec_patron] = f[idx_pico]

    print("\n[*] Generando graficas...")
    generar_graficas(resultados, archivos, dir_resultados)
    generar_grafica_aliasing(resultados, archivos, segmentos, dir_resultados)
    generar_grafica_psd(resultados, dir_resultados)
    generar_grafica_estabilidad(resultados, dir_resultados)
    generar_grafica_mantenimiento_calibracion(resultados, dir_resultados)
    generar_grafica_compresion_no_lineal_frecuencia(resultados, dir_resultados)

    print("\n[*] Exportando tablas CSV...")
    exportar_tabla_transferencia(resultados, dir_resultados)
    exportar_tabla_compresion(resultados, dir_resultados)
    exportar_tabla_estabilidad(resultados, dir_resultados)

    # --- Resumen de estabilidad en consola ---
    print("\n--- Estabilidad de frecuencia (tono patron) ---")
    for nombre, r in sorted(resultados.items()):
        for frec_patron, frec_detectada in r['estabilidad'].items():
            deriva = frec_detectada - frec_patron
            print(f"  [{nombre}] patron {frec_patron} Hz -> detectado {frec_detectada:.2f} Hz "
                  f"(deriva {deriva:+.2f} Hz)")

    print(f"\n[*] Listo. Graficas en: {dir_resultados}/")


# ==============================================================================
# GENERACION DE GRAFICAS
# ==============================================================================
def generar_graficas(resultados, archivos, dir_resultados):

    # Nivel de sweep usado para "retrato tonal": -12 dBFS evita zonas de
    # saturacion fuerte, asi la curva refleja color/EQ y no compresion.
    NIVEL_TONAL = -12

    def plot_freq_multi(claves_labels, nivel, titulo, archivo_salida, nota_txt, xlim=(20, 20000)):
        """claves_labels: lista de (nombre_archivo, etiqueta_leyenda). La
        primera entrada valida se usa como referencia (baseline) del panel
        de diferencia. El eje Y del panel principal se ajusta EXCLUSIVAMENTE
        con los valores que caen dentro de xlim (no con toda la curva, que
        puede incluir zonas fuera de rango que arruinan la escala)."""
        datos = []
        for nombre, label in claves_labels:
            if nombre is None or nombre not in resultados:
                continue
            fr = resultados[nombre]['freq_resp'].get(nivel)
            if fr is None:
                continue
            datos.append((label, fr))
        if not datos:
            print(f"    [!] Sin datos para: {titulo}")
            return

        con_delta = len(datos) >= 2
        if con_delta:
            fig, (ax, axd) = plt.subplots(
                2, 1, figsize=(8, 6.8), sharex=True,
                gridspec_kw={'height_ratios': [2.6, 1], 'hspace': 0.07})
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            axd = None

        valores_en_rango = []
        f_base, mag_base, label_base = None, None, None
        for i, (label, (f, mag)) in enumerate(datos):
            ax.plot(f, mag, label=label)
            mascara = (f >= xlim[0]) & (f <= xlim[1])
            if mascara.any():
                valores_en_rango.append(mag[mascara])
            if i == 0:
                f_base, mag_base, label_base = f, mag, label

        configurar_eje_frecuencia(ax, titulo, "Magnitud relativa (dB)", xlim=xlim)
        lim = limites_tight(np.concatenate(valores_en_rango)) if valores_en_rango else None
        if lim:
            ax.set_ylim(*lim)
        ax.legend(loc='lower left')

        if con_delta:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel('')
            deltas_en_rango = []
            for label, (f, mag) in datos[1:]:
                mag_interp = np.interp(f_base, f, mag)
                delta = mag_interp - mag_base
                axd.plot(f_base, delta, label=f"{label} − {label_base}")
                mascara = (f_base >= xlim[0]) & (f_base <= xlim[1])
                if mascara.any():
                    deltas_en_rango.append(delta[mascara])
            axd.axhline(0, color='k', linewidth=0.8, alpha=0.6)
            axd.set_xscale('log')
            axd.set_xlim(xlim)
            axd.set_xlabel('Frecuencia (Hz)')
            axd.set_ylabel('Diferencia (dB)')
            axd.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, pos: f"{int(x)}" if x < 1000 else f"{int(x/1000)}k"))
            axd.grid(True, which='both', linestyle='--', alpha=0.5)
            lim_d = limites_tight(np.concatenate(deltas_en_rango), pad_frac=0.15, pad_min=0.1) \
                if deltas_en_rango else None
            if lim_d:
                axd.set_ylim(*lim_d)
            axd.legend(fontsize=7, loc='best')

        fig.savefig(os.path.join(dir_resultados, archivo_salida), bbox_inches='tight')
        plt.close(fig)
        print(f"    [OK] {archivo_salida}")

    # -------------------------------------------------------------------
    # 01a. Retrato tonal Iron Deck (NAB 15 ips e IEC 30 ips)
    # -------------------------------------------------------------------
    iron_nom_15_nab = unico(resultados, es_referencia=False, ips=15.0, eq='NAB', rec=5.0)
    iron_nom_30_iec = unico(resultados, es_referencia=False, ips=30.0, eq='IEC', rec=5.0)
    plot_freq_multi(
        [(iron_nom_15_nab, "Iron Deck (NAB 15 ips)"),
         (iron_nom_30_iec, "Iron Deck (IEC 30 ips)")],
        NIVEL_TONAL,
        "Iron Deck: Retrato Tonal Nominal (NAB 15 ips vs IEC 30 ips)",
        "01a_Retrato_Tonal_IronDeck.png",
        "",
    )

    # -------------------------------------------------------------------
    # 01b. Retrato tonal Dispositivo de Referencia (NAB 15 ips y CCIR 15 ips)
    # -------------------------------------------------------------------
    ref_nom_15_nab = unico(resultados, es_referencia=True, ips=15.0, eq='NAB', input_db=0.0)
    ref_nom_15_ccir = unico(resultados, es_referencia=True, ips=15.0, eq='CCIR', input_db=0.0)
    plot_freq_multi(
        [(ref_nom_15_nab, "Referencia (NAB 15 ips)"),
         (ref_nom_15_ccir, "Referencia (CCIR 15 ips)")],
        NIVEL_TONAL,
        "Dispositivo de Referencia: Retrato Tonal Nominal (NAB vs CCIR)",
        "01b_Retrato_Tonal_Referencia.png",
        "",
    )

    # -------------------------------------------------------------------
    # 02. Efecto de la velocidad de cinta en Iron Deck (7.5 / 15 / 30 ips)
    # -------------------------------------------------------------------
    ips_disponibles = sorted({m['ips'] for m in archivos.values()
                               if not m['es_referencia'] and m['eq'] == 'NAB'})
    claves_vel = []
    for ips in ips_disponibles:
        k = unico(resultados, es_referencia=False, ips=ips, eq='NAB', rec=5.0)
        claves_vel.append((k, f"{ips:g} ips"))
    plot_freq_multi(
        claves_vel,
        NIVEL_TONAL,
        "Iron Deck: efecto de la velocidad de cinta (curva NAB)",
        "02_Efecto_Velocidad_Cinta_IronDeck.png",
        "",
    )

    # -------------------------------------------------------------------
    # 03. NAB vs IEC en Iron Deck a 30 ips
    # -------------------------------------------------------------------
    ips_con_ambas_eq = sorted({
        m['ips'] for m in archivos.values() if not m['es_referencia']
    } & {m['ips'] for m in archivos.values() if not m['es_referencia'] and m['eq'] == 'IEC'})
    if ips_con_ambas_eq:
        ips_ref = ips_con_ambas_eq[0]
        k_nab = unico(resultados, es_referencia=False, ips=ips_ref, eq='NAB', rec=5.0)
        k_iec = unico(resultados, es_referencia=False, ips=ips_ref, eq='IEC', rec=5.0)
        plot_freq_multi(
            [(k_nab, "NAB"), (k_iec, "IEC")],
            NIVEL_TONAL,
            f"Iron Deck: curva de reproduccion NAB vs IEC a {ips_ref:g} ips",
            "03_Efecto_Curva_EQ_IronDeck.png",
            "",
        )

    # -------------------------------------------------------------------
    # 04. Curva de saturacion/compresion estatica adaptada dinámicamente al rango medido
    # -------------------------------------------------------------------
    def plot_compresion(ax, claves_drive, frecuencia, colores):
        todos_x, todos_y = [], []
        for (nombre, label), color in zip(claves_drive, colores):
            if nombre not in resultados:
                continue
            datos = sorted(resultados[nombre]['compresion'].get(frecuencia, []))
            if len(datos) < 2:
                continue
            x_in = np.array([d[0] for d in datos])
            y_out = np.array([d[1] for d in datos])
            todos_x.append(x_in)
            todos_y.append(y_out)
            if len(x_in) >= 4:
                x_new = np.linspace(x_in.min(), x_in.max(), 300)
                spl = make_interp_spline(x_in, y_out, k=2)
                ax.plot(x_new, spl(x_new), label=label, color=color, linewidth=2)
            else:
                ax.plot(x_in, y_out, '-', label=label, color=color, linewidth=2)
            ax.plot(x_in, y_out, 'o', color=color, alpha=0.7, markersize=5)
        if todos_x and todos_y:
            concat_x = np.concatenate(todos_x)
            concat_y = np.concatenate(todos_y)
            min_v = min(concat_x.min(), concat_y.min())
            max_v = max(concat_x.max(), concat_y.max())
            pad = max((max_v - min_v) * 0.05, 0.5)
            lo, hi = min_v - pad, max_v + pad
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='Lineal 1:1')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel("Nivel de Entrada RMS (dBFS)", fontsize=10)
        ax.set_ylabel("Nivel de Salida RMS (dBFS)", fontsize=10)
        ax.set_title(f"Tono {frecuencia} Hz", fontweight='bold', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='upper left')

    ips_ref_comp = 15.0
    iron_drives = sorted(filtrar(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB'),
                          key=lambda k: resultados[k]['meta']['rec'])
    ref_drives = sorted(filtrar(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB'),
                         key=lambda k: resultados[k]['meta']['input_db'])
    if iron_drives or ref_drives:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        colores = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        claves_iron = [(k, f"Iron Deck {resultados[k]['meta']['drive_label']}") for k in iron_drives]
        claves_ref = [(k, f"Ref {resultados[k]['meta']['drive_label']}") for k in ref_drives]
        todas = claves_iron + claves_ref
        col_map = colores[:len(claves_iron)] + colores[:len(claves_ref)]
        plot_compresion(axes[0], todas, 1000, col_map)
        plot_compresion(axes[1], todas, 100, col_map)
        fig.suptitle(f"Curva de Transferencia Estática a {ips_ref_comp:g} ips (Adaptada al Rango RMS Medido)",
                     fontweight='bold', fontsize=13)
        fig.savefig(os.path.join(dir_resultados, "04_Curva_Saturacion_1kHz_100Hz.png"), bbox_inches='tight')
        plt.close(fig)
        print("    [OK] 04_Curva_Saturacion_1kHz_100Hz.png")
    else:
        print("    [!] Sin datos suficientes para curva de saturacion a 15 ips.")

    # -------------------------------------------------------------------
    # 05. THD vs nivel de entrada @ 1 kHz, por punto de drive (small
    #     multiples: Iron Deck | Referencia), a 15 ips.
    # -------------------------------------------------------------------
    if iron_drives or ref_drives:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        todos_y = []
        for ax, claves, titulo in [(axes[0], iron_drives, "Iron Deck"),
                                    (axes[1], ref_drives, "Referencia")]:
            for k, color in zip(claves, colores):
                datos = sorted(resultados[k]['thd'].get(1000, []))
                if not datos:
                    continue
                x = np.array([d[0] for d in datos], dtype=float)
                y = np.array([d[1] for d in datos], dtype=float)
                todos_y.append(y)
                ax.plot(x, y, 'o', color=color, alpha=0.6, markersize=4)
                # Interpolacion (spline cuadratico) sobre el nivel de
                # entrada para suavizar la curva entre los 5 puntos medidos.
                if len(x) >= 4:
                    x_new = np.linspace(x.min(), x.max(), 300)
                    spl = make_interp_spline(x, y, k=2)
                    y_smooth = np.maximum(spl(x_new), 0)
                    ax.plot(x_new, y_smooth, '-', color=color,
                            label=resultados[k]['meta']['drive_label'])
                else:
                    ax.plot(x, y, '-', color=color, label=resultados[k]['meta']['drive_label'])
            ax.set_xlabel("Nivel de entrada (dBFS)")
            ax.set_title(titulo, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
        if todos_y:
            lim = limites_tight(np.concatenate(todos_y), pad_frac=0.08, pad_min=0.05)
            if lim:
                axes[0].set_ylim(max(0, lim[0]), lim[1])
        axes[0].set_ylabel("THD (%)")
        fig.suptitle(f"Distorsion armonica total vs nivel, tono 1 kHz, {ips_ref_comp:g} ips", fontweight='bold')
        fig.savefig(os.path.join(dir_resultados, "05_THD_vs_Nivel.png"), bbox_inches='tight')
        plt.close(fig)
        print("    [OK] 05_THD_vs_Nivel.png")

    # Las graficas 06 (aliasing), 07 (PSD) y 08 (estabilidad) se generan
    # aparte porque 06 necesita releer WAVs crudos usando los segmentos
    # del indice; ver generar_grafica_aliasing / generar_grafica_psd /
    # generar_grafica_estabilidad, todas llamadas desde analizar_proyecto().


def generar_grafica_aliasing(resultados, archivos, segmentos, dir_resultados, ips_ref_comp=15.0):
    """Grafica 06: espectro del tono de 5 kHz al maximo drive disponible,
    para detectar artefactos de aliasing (picos fuera de los multiplos de 5 kHz)."""
    def cargar_wav_por_nombre(nombre):
        for ruta in archivos:
            if os.path.basename(ruta) == nombre:
                return ruta
        return None

    iron_max = sorted(filtrar(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB'),
                       key=lambda k: -resultados[k]['meta']['rec'])
    ref_max = sorted(filtrar(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB'),
                      key=lambda k: -resultados[k]['meta']['input_db'])
    if not iron_max or not ref_max:
        print("    [!] Sin datos suficientes para prueba de aliasing.")
        return

    seg_5k = [s for s in segmentos if "5000 Hz" in s['etiqueta'] and "+0 dBFS" in s['etiqueta']]
    if not seg_5k:
        print("    [!] No se encontro el segmento de tono 5 kHz a 0 dBFS en el indice.")
        return
    seg_5k = seg_5k[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    XLIM_ALIAS = (20, 24000)
    valores_en_rango = []

    def plot_una(nombre, label, color):
        ruta = cargar_wav_por_nombre(nombre)
        if ruta is None:
            return
        fs_w, a_wet = wavfile.read(ruta)
        if a_wet.ndim > 1:
            a_wet = a_wet[:, 0]
        senal = aislar_senal(a_wet, fs_w, seg_5k['inicio'], seg_5k['fin'])
        if len(senal) == 0:
            return
        f, Pxx = signal.welch(senal, fs_w, nperseg=min(len(senal), fs_w * 2))
        mag_db = 10 * np.log10(Pxx + 1e-12)
        idx_fund = np.argmin(np.abs(f - 5000))
        mag_db = mag_db - mag_db[idx_fund]
        ax.plot(f, mag_db, label=label, color=color, alpha=0.85)
        mascara = (f >= XLIM_ALIAS[0]) & (f <= XLIM_ALIAS[1])
        if mascara.any():
            valores_en_rango.append(mag_db[mascara])

    meta_iron = resultados[iron_max[0]]['meta']
    meta_ref = resultados[ref_max[0]]['meta']
    plot_una(iron_max[0], f"Iron Deck ({meta_iron['drive_label']})", 'tab:orange')
    plot_una(ref_max[0], f"Referencia ({meta_ref['drive_label']})", 'tab:blue')

    ax.set_xscale('log')
    ax.set_xlim(*XLIM_ALIAS)
    # Escalado ajustado a los datos reales (no un rango fijo -140..5 dB que
    # deja la mitad del grafico vacio si el piso de ruido no llega tan bajo).
    lim = limites_tight(np.concatenate(valores_en_rango), pad_frac=0.05, pad_min=2.0) \
        if valores_en_rango else None
    if lim:
        ax.set_ylim(lim[0], max(lim[1], 3.0))
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud relativa (dB, normalizada a la fundamental)')
    ax.set_title(f'Prueba de aliasing: tono 5 kHz a saturacion maxima ({ips_ref_comp:g} ips)', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x < 1000 else f"{int(x/1000)}k"))
    for k in range(1, 6):
        if k * 5000 <= 24000:
            ax.axvline(k * 5000, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    fig.savefig(os.path.join(dir_resultados, "06_Aliasing_5kHz_Saturacion_Extrema.png"), bbox_inches='tight')
    plt.close(fig)
    print("    [OK] 06_Aliasing_5kHz_Saturacion_Extrema.png")


def generar_grafica_psd(resultados, dir_resultados, ips_ref_comp=15.0, nivel=-12):
    """Grafica 07: distribucion de ENERGIA POR BANDA DE OCTAVA del ruido
    rosa (Seccion IV-D), no una curva continua punto a punto. Dos paneles:
      (a) Iron Deck vs Referencia, ajuste nominal -> donde concentra cada
          uno su coloracion caracteristica.
      (b) Iron Deck a drive minimo vs maximo -> como migra esa energia al
          empujar la saturacion (lo que pide la Seccion IV-D del documento).
    Cada panel de barras trae debajo su propio panel de diferencia banda a
    banda, ajustado a la escala real de esa diferencia (asi se detectan
    cambios de decimas de dB que en una curva superpuesta pasan desapercibidos).
    """
def generar_grafica_psd(resultados, dir_resultados, ips_ref_comp=15.0, nivel=-12):
    """
    Grafica 07: Distribución de ENERGIA POR BANDA DE OCTAVA (Ruido Rosa).
    Gráfica independiente, ampliada y de alta legibilidad.
    """
    iron_nom = unico(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB', rec=5.0)
    ref_nom = unico(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB', input_db=0.0)
    iron_min = unico(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB', rec=0.0)
    iron_max = unico(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB', rec=10.0)

    paneles = []
    if iron_nom and ref_nom and nivel in resultados[iron_nom]['psd'] and nivel in resultados[ref_nom]['psd']:
        paneles.append(('dispositivos', iron_nom, ref_nom))
    if iron_min and iron_max and nivel in resultados[iron_min]['psd'] and nivel in resultados[iron_max]['psd']:
        paneles.append(('drive', iron_max, iron_min))

    if not paneles:
        print("    [!] Sin datos suficientes para el analisis por bandas de ruido rosa.")
        return

    fig, axes = plt.subplots(2, len(paneles), figsize=(7.5 * len(paneles), 8.5),
                              gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.15})
    if len(paneles) == 1:
        axes = axes.reshape(2, 1)

    for col, (tipo, clave_a, clave_b) in enumerate(paneles):
        f, mag_a = resultados[clave_a]['psd'][nivel]
        _, mag_b = resultados[clave_b]['psd'][nivel]
        bandas_a = calcular_bandas_octava(f, mag_a)
        bandas_b = calcular_bandas_octava(f, mag_b)
        if tipo == 'dispositivos':
            label_a, label_b = "Iron Deck (NAB)", "Dispositivo de Referencia (NAB)"
            titulo = (f"Iron Deck vs Referencia (Ajuste Nominal)\n"
                      f"Ruido rosa {nivel:+d} dBFS, {ips_ref_comp:g} ips")
        else:
            label_a = f"Iron Deck ({resultados[clave_a]['meta']['drive_label']})"
            label_b = f"Iron Deck ({resultados[clave_b]['meta']['drive_label']})"
            titulo = (f"Iron Deck: Migración Espectral con el Drive\n"
                      f"Ruido rosa {nivel:+d} dBFS, {ips_ref_comp:g} ips")
        plot_comparacion_bandas(axes[0, col], axes[1, col], CENTROS_OCTAVA,
                                 bandas_a, bandas_b, label_a, label_b, titulo)

    fig.suptitle("Distribución de Energía por Bandas de Octava (Ruido Rosa)", fontweight='bold', fontsize=14, y=0.98)
    fig.savefig(os.path.join(dir_resultados, "07_Distribucion_Espectral_RuidoRosa.png"), bbox_inches='tight')
    plt.close(fig)
    print("    [OK] 07_Distribucion_Espectral_RuidoRosa.png")


def generar_grafica_estabilidad(resultados, dir_resultados):
    """
    Grafica 08: Prueba de Estabilidad de Frecuencia Digital y Calibración.
    Gráfica independiente, ampliada y clasificada por:
      - Estándar de calibración (NAB: 3000 Hz vs DIN/IEC: 3150 Hz)
      - Comparativa directa entre ambos plugins.
    """
    filas_nab = []
    filas_iec = []
    for nombre, r in resultados.items():
        meta = r['meta']
        disp = "Referencia" if meta['es_referencia'] else "Iron Deck"
        lbl = f"{disp} {meta['ips']:g}ips {meta['eq']} ({meta['drive_label']})"
        for frec_patron, frec_detectada in r['estabilidad'].items():
            deriva = frec_detectada - frec_patron
            if frec_patron == 3000:
                filas_nab.append((lbl, disp, deriva))
            elif frec_patron == 3150:
                filas_iec.append((lbl, disp, deriva))

    if not filas_nab and not filas_iec:
        print("    [!] Sin datos de estabilidad de frecuencia.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)

    def plot_sub_estabilidad(ax, filas, titulo_std, frec_target):
        filas.sort(key=lambda x: x[0])
        etiquetas = [f[0] for f in filas]
        derivas = [f[2] for f in filas]
        colores = ['#1f77b4' if 'Iron Deck' in f[1] else '#ff7f0e' for f in filas]

        ax.barh(etiquetas, derivas, color=colores, alpha=0.85, height=0.6)
        ax.axvline(0, color='k', linewidth=1, linestyle='--')
        ax.set_xlabel(f"Deriva de Frecuencia (Hz) vs Tono Patrón {frec_target} Hz", fontsize=10)
        ax.set_title(titulo_std, fontweight='bold', fontsize=12)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(-1.0, 1.0)
        for i, val in enumerate(derivas):
            ax.text(val + (0.05 if val >= 0 else -0.2), i, f"{val:+.2f} Hz", va='center', fontsize=8, fontweight='bold')

    plot_sub_estabilidad(axes[0], filas_nab, "Estándar NAB (Tono Patrón 3000 Hz)", 3000)
    plot_sub_estabilidad(axes[1], filas_iec, "Estándar DIN/IEC (Tono Patrón 3150 Hz)", 3150)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Iron Deck (M Media Audio)'),
        Patch(facecolor='#ff7f0e', label='Dispositivo de Referencia')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)

    fig.suptitle("Prueba de Estabilidad Digital y Calibración Clasificada por Estándar de Frecuencia", fontweight='bold', fontsize=14, y=1.03)
    fig.savefig(os.path.join(dir_resultados, "08_Estabilidad_Frecuencia.png"), bbox_inches='tight')
    plt.close(fig)
    print("    [OK] 08_Estabilidad_Frecuencia.png")


def generar_grafica_mantenimiento_calibracion(resultados, dir_resultados, ips_ref_comp=15.0, nivel=-12):
    """
    Grafica 09: Evaluacion del MANTENIMIENTO DE LA CALIBRACION en funcion de la ganancia/drive.
    Compara H(f) a traves de los puntos de drive para Iron Deck y Referencia, e aisla la
    deriva tonal relativa ΔH(f) = H(f, Drive) - H(f, Drive_nom).
    """
    iron_keys = sorted(filtrar(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB'),
                       key=lambda k: resultados[k]['meta']['rec'])
    ref_keys = sorted(filtrar(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB'),
                      key=lambda k: resultados[k]['meta']['input_db'])

    if not iron_keys or not ref_keys:
        print("    [!] Sin datos suficientes para evaluacion de calibracion vs ganancia.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.08})

    colores_iron = ['#1f77b4', '#2ca02c', '#d62728']
    colores_ref = ['#1f77b4', '#2ca02c', '#d62728']
    xlim = (20, 20000)

    # Subplot 1: Iron Deck H(f) por Drive
    base_iron_k = iron_keys[len(iron_keys)//2]
    f_base_i, mag_base_i = resultados[base_iron_k]['freq_resp'][nivel]

    val_i = []
    for k, color in zip(iron_keys, colores_iron):
        f, mag = resultados[k]['freq_resp'][nivel]
        lbl = f"Iron Deck {resultados[k]['meta']['drive_label']}"
        axes[0, 0].plot(f, mag, label=lbl, color=color)
        m = (f >= xlim[0]) & (f <= xlim[1])
        val_i.append(mag[m])
        delta = mag - np.interp(f, f_base_i, mag_base_i)
        axes[1, 0].plot(f, delta, label=f"Δ {resultados[k]['meta']['drive_label']}", color=color)

    configurar_eje_frecuencia(axes[0, 0], f"Iron Deck: Respuesta por Drive REC ({ips_ref_comp:g} ips NAB)", "Magnitud (dB)", xlim=xlim)
    axes[0, 0].legend(fontsize=8)
    lim_i = limites_tight(np.concatenate(val_i))
    if lim_i:
        axes[0, 0].set_ylim(*lim_i)

    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.6)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_xlabel('Frecuencia (Hz)')
    axes[1, 0].set_ylabel('Deriva Calibración Δ (dB)')
    axes[1, 0].grid(True, which='both', linestyle='--', alpha=0.5)

    # Subplot 2: Referencia H(f) por Drive
    base_ref_k = ref_keys[len(ref_keys)//2]
    f_base_r, mag_base_r = resultados[base_ref_k]['freq_resp'][nivel]

    val_r = []
    for k, color in zip(ref_keys, colores_ref):
        f, mag = resultados[k]['freq_resp'][nivel]
        lbl = f"Ref {resultados[k]['meta']['drive_label']}"
        axes[0, 1].plot(f, mag, label=lbl, color=color)
        m = (f >= xlim[0]) & (f <= xlim[1])
        val_r.append(mag[m])
        delta = mag - np.interp(f, f_base_r, mag_base_r)
        axes[1, 1].plot(f, delta, label=f"Δ {resultados[k]['meta']['drive_label']}", color=color)

    configurar_eje_frecuencia(axes[0, 1], f"Referencia: Respuesta por Input ({ips_ref_comp:g} ips NAB)", "Magnitud (dB)", xlim=xlim)
    axes[0, 1].legend(fontsize=8)
    lim_r = limites_tight(np.concatenate(val_r))
    if lim_r:
        axes[0, 1].set_ylim(*lim_r)

    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.6)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlim(xlim)
    axes[1, 1].set_xlabel('Frecuencia (Hz)')
    axes[1, 1].set_ylabel('Deriva Calibración Δ (dB)')
    axes[1, 1].grid(True, which='both', linestyle='--', alpha=0.5)

    fig.suptitle("Evaluación de Mantenimiento de Calibración Frente al Aumento de Ganancia", fontweight='bold', fontsize=13)
    fig.savefig(os.path.join(dir_resultados, "09_Mantenimiento_Calibracion_vs_Ganancia.png"), bbox_inches='tight')
    plt.close(fig)
    print("    [OK] 09_Mantenimiento_Calibracion_vs_Ganancia.png")


def generar_grafica_compresion_no_lineal_frecuencia(resultados, dir_resultados, ips_ref_comp=15.0):
    """
    Grafica 10: Comparativa de compresión no lineal DEPENDIENTE DE LA FRECUENCIA.
    Ejes X e Y adaptados exactamente al rango dinámico RMS obtenido (100 Hz vs 1000 Hz vs 5000 Hz).
    """
    iron_nom = unico(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB', rec=5.0)
    ref_nom = unico(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB', input_db=0.0)

    if not iron_nom or not ref_nom:
        print("    [!] Sin datos para analisis de compresion por frecuencia.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.15})
    frecuencias = [100, 1000, 5000]
    colores_frec = {'100': '#1f77b4', '1000': '#2ca02c', '5000': '#d62728'}

    for idx, (clave, titulo) in enumerate([(iron_nom, "Iron Deck (Ajuste Nominal)"),
                                            (ref_nom, "Dispositivo de Referencia (Ajuste Nominal)")]):
        todos_x, todos_y = [], []
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        for frec in frecuencias:
            datos = sorted(resultados[clave]['compresion'].get(frec, []))
            if not datos:
                continue
            x_in = np.array([d[0] for d in datos])
            y_out = np.array([d[1] for d in datos])
            todos_x.append(x_in)
            todos_y.append(y_out)
            color = colores_frec[str(frec)]

            if len(x_in) >= 4:
                x_new = np.linspace(x_in.min(), x_in.max(), 300)
                spl = make_interp_spline(x_in, y_out, k=2)
                ax_top.plot(x_new, spl(x_new), '-', label=f"{frec} Hz", color=color, linewidth=2)
                delta_y = y_out - x_in
                spl_d = make_interp_spline(x_in, delta_y, k=2)
                ax_bot.plot(x_new, spl_d(x_new), '-', label=f"{frec} Hz", color=color, linewidth=1.8)
            else:
                ax_top.plot(x_in, y_out, '-', label=f"{frec} Hz", color=color, linewidth=2)
                ax_bot.plot(x_in, y_out - x_in, '-', label=f"{frec} Hz", color=color, linewidth=1.8)
            ax_top.plot(x_in, y_out, 'o', color=color, alpha=0.7, markersize=5)
            ax_bot.plot(x_in, y_out - x_in, 'o', color=color, alpha=0.7, markersize=4)

        if todos_x and todos_y:
            concat_x = np.concatenate(todos_x)
            concat_y = np.concatenate(todos_y)
            min_v = min(concat_x.min(), concat_y.min())
            max_v = max(concat_x.max(), concat_y.max())
            pad = max((max_v - min_v) * 0.05, 0.5)
            lo, hi = min_v - pad, max_v + pad
            ax_top.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='Lineal 1:1')
            ax_top.set_xlim(lo, hi)
            ax_top.set_ylim(lo, hi)
            ax_bot.set_xlim(lo, hi)

        ax_top.set_ylabel("Nivel de Salida RMS (dBFS)", fontsize=10)
        ax_top.set_title(titulo, fontweight='bold', fontsize=12)
        ax_top.grid(True, linestyle='--', alpha=0.5)
        ax_top.legend(fontsize=9, loc='upper left')

        ax_bot.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_bot.set_xlabel("Nivel de Entrada RMS (dBFS)", fontsize=10)
        ax_bot.set_ylabel("Compresión / Delta RMS (dB)", fontsize=9)
        ax_bot.grid(True, linestyle='--', alpha=0.5)
        ax_bot.legend(fontsize=8, loc='best')

    fig.suptitle("Análisis de Compresión y Saturación No Lineal por Banda de Frecuencia", fontweight='bold', fontsize=13)
    fig.savefig(os.path.join(dir_resultados, "10_Compresion_NoLineal_Frecuencia.png"), bbox_inches='tight')
    plt.close(fig)
    print("    [OK] 10_Compresion_NoLineal_Frecuencia.png")


# ==============================================================================
# EXPORTADORES DE TABLAS CSV
# ==============================================================================
def exportar_tabla_transferencia(resultados, dir_resultados, ips_ref_comp=15.0):
    """
    Tabla 04-CSV: Curva de Transferencia Estática (Input RMS dBFS vs Output RMS dBFS)
    para los tonos 1 kHz y 100 Hz, separada por plugin y punto de drive.
    """
    import csv
    ruta = os.path.join(dir_resultados, "04_tabla_transferencia_estatica.csv")
    iron_drives = sorted(filtrar(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB'),
                         key=lambda k: resultados[k]['meta']['rec'])
    ref_drives = sorted(filtrar(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB'),
                        key=lambda k: resultados[k]['meta']['input_db'])
    filas = []
    for frecuencia in [100, 1000]:
        for nombre in iron_drives + ref_drives:
            meta = resultados[nombre]['meta']
            dispositivo = "Iron Deck" if not meta['es_referencia'] else "Referencia"
            drive_lbl = meta['drive_label']
            datos = sorted(resultados[nombre]['compresion'].get(frecuencia, []))
            for db_in, db_out in datos:
                compresion = db_out - db_in
                filas.append({
                    'Dispositivo': dispositivo,
                    'Drive': drive_lbl,
                    'Frecuencia_Hz': frecuencia,
                    'Input_dBFS': f"{db_in:.2f}",
                    'Output_dBFS': f"{db_out:.2f}",
                    'Compresion_neta_dB': f"{compresion:.2f}",
                })
    if not filas:
        print("    [!] Sin datos para tabla de transferencia estática.")
        return
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    print(f"    [OK] {os.path.basename(ruta)}")


def exportar_tabla_compresion(resultados, dir_resultados, ips_ref_comp=15.0):
    """
    Tabla 10-CSV: Compresión no lineal dependiente de la frecuencia en ajuste nominal.
    Calcula la compresión neta (Output - Input) para 100 Hz, 1 kHz y 5 kHz.
    """
    import csv
    ruta = os.path.join(dir_resultados, "10_tabla_compresion_frecuencia.csv")
    iron_nom = unico(resultados, es_referencia=False, ips=ips_ref_comp, eq='NAB', rec=5.0)
    ref_nom = unico(resultados, es_referencia=True, ips=ips_ref_comp, eq='NAB', input_db=0.0)
    filas = []
    for clave, dispositivo in [(iron_nom, "Iron Deck"), (ref_nom, "Referencia")]:
        if clave is None:
            continue
        for frecuencia in [100, 1000, 5000]:
            datos = sorted(resultados[clave]['compresion'].get(frecuencia, []))
            for db_in, db_out in datos:
                compresion = db_out - db_in
                filas.append({
                    'Dispositivo': dispositivo,
                    'Frecuencia_Hz': frecuencia,
                    'Input_dBFS': f"{db_in:.2f}",
                    'Output_dBFS': f"{db_out:.2f}",
                    'Compresion_neta_dB': f"{compresion:.2f}",
                })
    if not filas:
        print("    [!] Sin datos para tabla de compresión por frecuencia.")
        return
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    print(f"    [OK] {os.path.basename(ruta)}")


def exportar_tabla_estabilidad(resultados, dir_resultados):
    """
    Tabla 08-CSV: Prueba de Estabilidad Digital de Frecuencia.
    Clasifica por estándar de calibración (NAB 3000 Hz / DIN-IEC 3150 Hz)
    y compara entre plugins.
    """
    import csv
    ruta = os.path.join(dir_resultados, "08_tabla_estabilidad_frecuencia.csv")
    filas = []
    for nombre, r in sorted(resultados.items()):
        meta = r['meta']
        dispositivo = "Referencia" if meta['es_referencia'] else "Iron Deck"
        eq = meta['eq']
        ips = meta['ips']
        drive_lbl = meta['drive_label']
        for frec_patron, frec_detectada in r['estabilidad'].items():
            deriva = frec_detectada - frec_patron
            if frec_patron == 3000:
                estandar = "NAB"
            elif frec_patron == 3150:
                estandar = "DIN/IEC"
            else:
                estandar = str(frec_patron)
            filas.append({
                'Dispositivo': dispositivo,
                'EQ': eq,
                'IPS': f"{ips:g}",
                'Drive': drive_lbl,
                'Estandar_Calibracion': estandar,
                'Tono_Patron_Hz': frec_patron,
                'Frecuencia_Detectada_Hz': f"{frec_detectada:.2f}",
                'Deriva_Hz': f"{deriva:+.2f}",
            })
    if not filas:
        print("    [!] Sin datos de estabilidad para exportar.")
        return
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    print(f"    [OK] {os.path.basename(ruta)}")


if __name__ == "__main__":
    import sys
    dir_proyecto = sys.argv[1] if len(sys.argv) > 1 else "."
    analizar_proyecto(dir_proyecto=dir_proyecto, dir_resultados="resultados_analisis")
