"""
Projeto A3 - Processamento Digital de Imagens
Tema: Inspeção visual automatizada de equipamentos de combate a incêndio
       (abrigo de hidrante de parede) como ferramenta auxiliar de gestão
       ambiental urbana.

Pipeline:
    1. Carregamento e pré-processamento (BGR->RGB, cinza, redimensionamento, normalização)
    2. Análise da imagem original (metadados + histograma RGB)
    3. Filtros (Gaussiano, Mediana, Bilateral)
    4. Detecção de bordas (Canny) - cinza e sobre máscara HSV
    5. Segmentação (Otsu e HSV por cor vermelha)
    6. Morfologia (erosão + dilatação + abertura/fechamento)
    7. Contornos sobrepostos
    8. Métricas (% área, média, desvio, PSNR, SNR)
    9. Grid comparativo final

Uso:
    python processamento.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio


# ---------------------------------------------------------------------------
# Constantes do projeto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
CAMINHO_IMAGEM = RAIZ / "imagens" / "imagem_original.jpg"
PASTA_RESULTADOS = RAIZ / "resultados"

LARGURA_ALVO = 600  # imagem em retrato; 600 px de largura -> ~800 px de altura

# Faixas HSV para o vermelho do hidrante (vermelho cruza H=0 -> duas faixas)
# H limitado a 10 na faixa baixa para excluir o laranja dos azulejos (H~10-20)
HSV_VERMELHO_BAIXO_1 = np.array([0, 120, 60], dtype=np.uint8)
HSV_VERMELHO_ALTO_1 = np.array([10, 255, 255], dtype=np.uint8)
HSV_VERMELHO_BAIXO_2 = np.array([170, 120, 60], dtype=np.uint8)
HSV_VERMELHO_ALTO_2 = np.array([179, 255, 255], dtype=np.uint8)

KERNEL_MORFOLOGIA = np.ones((5, 5), np.uint8)
AREA_MINIMA_CONTORNO = 500

CANNY_T1 = 50
CANNY_T2 = 150


# ---------------------------------------------------------------------------
# 1. Carregamento e pré-processamento
# ---------------------------------------------------------------------------
def carregar_imagem(caminho: Path) -> np.ndarray:
    """Lê uma imagem do disco e converte de BGR (OpenCV) para RGB."""
    bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocessar(img_rgb: np.ndarray, largura_alvo: int = LARGURA_ALVO) -> dict:
    """Redimensiona, converte para cinza e normaliza.

    Retorna dicionário com:
        - rgb: imagem RGB uint8 redimensionada
        - gray: imagem em escala de cinza uint8 redimensionada
        - rgb_norm: imagem RGB normalizada em [0, 1] float32
    """
    altura_orig, largura_orig = img_rgb.shape[:2]
    escala = largura_alvo / largura_orig
    nova_altura = int(altura_orig * escala)
    rgb = cv2.resize(img_rgb, (largura_alvo, nova_altura), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_norm = rgb.astype(np.float32) / 255.0
    return {"rgb": rgb, "gray": gray, "rgb_norm": rgb_norm}


# ---------------------------------------------------------------------------
# 2. Análise da imagem original (metadados + histograma)
# ---------------------------------------------------------------------------
def imprimir_metadados(img_rgb: np.ndarray, caminho: Path) -> dict:
    """Coleta e imprime metadados básicos da imagem."""
    altura, largura, canais = img_rgb.shape
    tamanho_kb = os.path.getsize(caminho) / 1024
    meta = {
        "resolucao": f"{largura} x {altura}",
        "canais": canais,
        "tamanho_kb": round(tamanho_kb, 2),
        "dtype": str(img_rgb.dtype),
    }
    print("== Metadados da imagem original ==")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return meta


def gerar_histograma_rgb(img_rgb: np.ndarray, caminho_saida: Path) -> None:
    """Plota e salva o histograma dos três canais RGB."""
    cores = ("red", "green", "blue")
    nomes = ("R", "G", "B")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cor, nome) in enumerate(zip(cores, nomes)):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax.plot(hist, color=cor, label=f"Canal {nome}")
    ax.set_title("Histograma dos canais R, G, B")
    ax.set_xlabel("Intensidade (0-255)")
    ax.set_ylabel("Frequência de pixels")
    ax.set_xlim(0, 255)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Filtros
# ---------------------------------------------------------------------------
def aplicar_filtros(img_rgb: np.ndarray) -> dict:
    """Aplica Gaussiano (sigma=1.5), Mediana (5x5) e Bilateral."""
    gauss = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=1.5)
    mediana = cv2.medianBlur(img_rgb, 5)
    bilateral = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    return {"gaussiano": gauss, "mediana": mediana, "bilateral": bilateral}


# ---------------------------------------------------------------------------
# 4. Detecção de bordas
# ---------------------------------------------------------------------------
def detectar_bordas(gray: np.ndarray, t1: int = CANNY_T1, t2: int = CANNY_T2) -> np.ndarray:
    """Aplica suavização Gaussiana leve e em seguida o operador Canny."""
    suavizada = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(suavizada, t1, t2)


# ---------------------------------------------------------------------------
# 5. Segmentação
# ---------------------------------------------------------------------------
def segmentar_otsu(gray: np.ndarray) -> tuple[float, np.ndarray]:
    """Segmentação por limiarização de Otsu."""
    limiar, mascara = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return limiar, mascara


def segmentar_hsv_vermelho(img_rgb: np.ndarray) -> np.ndarray:
    """Segmentação por cor vermelha no espaço HSV (duas faixas em H)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, HSV_VERMELHO_BAIXO_1, HSV_VERMELHO_ALTO_1)
    m2 = cv2.inRange(hsv, HSV_VERMELHO_BAIXO_2, HSV_VERMELHO_ALTO_2)
    return cv2.bitwise_or(m1, m2)


# ---------------------------------------------------------------------------
# 6. Morfologia
# ---------------------------------------------------------------------------
def aplicar_morfologia(mascara: np.ndarray) -> dict:
    """Aplica erosão, dilatação e a combinação abertura+fechamento."""
    erosao = cv2.erode(mascara, KERNEL_MORFOLOGIA, iterations=1)
    dilatacao = cv2.dilate(mascara, KERNEL_MORFOLOGIA, iterations=1)
    abertura = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL_MORFOLOGIA)
    final = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, KERNEL_MORFOLOGIA)
    return {
        "erosao": erosao,
        "dilatacao": dilatacao,
        "abertura": abertura,
        "final": final,
    }


# ---------------------------------------------------------------------------
# 7. Contornos
# ---------------------------------------------------------------------------
def desenhar_contornos(
    img_rgb: np.ndarray, mascara: np.ndarray, area_min: int = AREA_MINIMA_CONTORNO
) -> tuple[np.ndarray, int]:
    """Desenha em verde os contornos cuja área >= area_min."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtrados = [c for c in contornos if cv2.contourArea(c) >= area_min]
    saida = img_rgb.copy()
    cv2.drawContours(saida, filtrados, -1, (0, 255, 0), 3)
    return saida, len(filtrados)


# ---------------------------------------------------------------------------
# 8. Métricas
# ---------------------------------------------------------------------------
def calcular_snr(original: np.ndarray, filtrada: np.ndarray) -> float:
    """SNR em dB: 10*log10(P_sinal / P_ruido), onde ruido = original - filtrada."""
    orig_f = original.astype(np.float64)
    filt_f = filtrada.astype(np.float64)
    ruido = orig_f - filt_f
    p_sinal = np.mean(orig_f ** 2)
    p_ruido = np.mean(ruido ** 2)
    if p_ruido < 1e-12:
        return float("inf")
    return 10.0 * np.log10(p_sinal / p_ruido)


def calcular_metricas(
    rgb: np.ndarray,
    gray: np.ndarray,
    filtros: dict,
    mascara_segmentacao: np.ndarray,
) -> pd.DataFrame:
    """Calcula percentual de área segmentada, média/desvio e PSNR/SNR por filtro."""
    total_pixels = mascara_segmentacao.size
    pixels_segmentados = int(np.count_nonzero(mascara_segmentacao))
    pct_area = 100.0 * pixels_segmentados / total_pixels

    linhas = [
        ("Pixels totais", total_pixels),
        ("Pixels segmentados (HSV+morfologia)", pixels_segmentados),
        ("% de area segmentada", round(pct_area, 2)),
        ("Valor medio de pixel (cinza)", round(float(gray.mean()), 2)),
        ("Desvio-padrao (cinza)", round(float(gray.std()), 2)),
    ]
    for nome, filtrada in filtros.items():
        psnr = peak_signal_noise_ratio(rgb, filtrada, data_range=255)
        snr = calcular_snr(rgb, filtrada)
        linhas.append((f"PSNR {nome} (dB)", round(float(psnr), 2)))
        linhas.append((f"SNR {nome} (dB)", round(float(snr), 2)))

    return pd.DataFrame(linhas, columns=["metrica", "valor"])


# ---------------------------------------------------------------------------
# 9. Salvamento e grid final
# ---------------------------------------------------------------------------
def salvar_imagem(img: np.ndarray, caminho: Path, cmap: str | None = None) -> None:
    """Salva uma imagem (RGB ou binária) como PNG usando matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)


def montar_grid(paineis: list, caminho_saida: Path) -> None:
    """Monta um grid 3x3 com nove painéis (título, imagem, cmap)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 16))
    for ax, (titulo, img, cmap) in zip(axes.flat, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def main() -> None:
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    # --- Fase 5: carregar e pré-processar
    original_rgb = carregar_imagem(CAMINHO_IMAGEM)
    pre = preprocessar(original_rgb)
    rgb, gray = pre["rgb"], pre["gray"]
    salvar_imagem(rgb, PASTA_RESULTADOS / "01_original.png")
    salvar_imagem(gray, PASTA_RESULTADOS / "02_cinza.png", cmap="gray")

    # --- Fase 6: metadados + histograma
    imprimir_metadados(original_rgb, CAMINHO_IMAGEM)
    gerar_histograma_rgb(rgb, PASTA_RESULTADOS / "03_histograma_rgb.png")

    # --- Fase 7: filtros
    filtros = aplicar_filtros(rgb)
    salvar_imagem(filtros["gaussiano"], PASTA_RESULTADOS / "04_gaussiano.png")
    salvar_imagem(filtros["mediana"], PASTA_RESULTADOS / "05_mediana.png")
    salvar_imagem(filtros["bilateral"], PASTA_RESULTADOS / "06_bilateral.png")

    # --- Fase 8: bordas
    bordas_cinza = detectar_bordas(gray)
    salvar_imagem(bordas_cinza, PASTA_RESULTADOS / "07_canny.png", cmap="gray")

    # --- Fase 9: segmentação Otsu e HSV
    limiar_otsu, mascara_otsu = segmentar_otsu(gray)
    print(f"Limiar Otsu calculado: {limiar_otsu:.1f}")
    salvar_imagem(mascara_otsu, PASTA_RESULTADOS / "08_otsu.png", cmap="gray")

    mascara_hsv = segmentar_hsv_vermelho(rgb)
    salvar_imagem(mascara_hsv, PASTA_RESULTADOS / "09_mascara_hsv.png", cmap="gray")

    # --- Morfologia sobre a máscara HSV (principal)
    morf = aplicar_morfologia(mascara_hsv)
    salvar_imagem(morf["erosao"], PASTA_RESULTADOS / "10a_erosao.png", cmap="gray")
    salvar_imagem(morf["dilatacao"], PASTA_RESULTADOS / "10b_dilatacao.png", cmap="gray")
    salvar_imagem(morf["final"], PASTA_RESULTADOS / "10_morfologia.png", cmap="gray")

    # --- Canny sobre a máscara morfológica (mostra ganho vs. Canny cinza)
    bordas_mascara = cv2.Canny(morf["final"], CANNY_T1, CANNY_T2)
    salvar_imagem(bordas_mascara, PASTA_RESULTADOS / "07b_canny_mascara.png", cmap="gray")

    # --- Contornos
    contornos_img, n_contornos = desenhar_contornos(rgb, morf["final"])
    print(f"Contornos encontrados (area >= {AREA_MINIMA_CONTORNO}): {n_contornos}")
    salvar_imagem(contornos_img, PASTA_RESULTADOS / "11_contornos.png")

    # --- Fase 10: métricas
    metricas = calcular_metricas(rgb, gray, filtros, morf["final"])
    metricas.to_csv(PASTA_RESULTADOS / "metricas.csv", index=False)
    print("\n== Metricas ==")
    print(metricas.to_string(index=False))

    # --- Fase 11: grid final 3x3
    paineis = [
        ("Original (RGB)", rgb, None),
        ("Escala de cinza", gray, "gray"),
        ("Bilateral", filtros["bilateral"], None),
        ("Gaussiano (sigma=1.5)", filtros["gaussiano"], None),
        ("Mediana (5x5)", filtros["mediana"], None),
        ("Canny (cinza)", bordas_cinza, "gray"),
        ("Otsu", mascara_otsu, "gray"),
        ("Mascara HSV + morfologia", morf["final"], "gray"),
        ("Contornos sobrepostos", contornos_img, None),
    ]
    montar_grid(paineis, PASTA_RESULTADOS / "12_grid_final.png")

    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
