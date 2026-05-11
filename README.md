# Projeto A3 — Processamento Digital de Imagens

**Tema:** Inspeção visual automatizada de equipamentos de combate a incêndio (abrigo de hidrante de parede) como ferramenta auxiliar de gestão ambiental urbana.

## Recorte ambiental

Incêndios urbanos são vetores significativos de degradação ambiental: emitem CO₂ e material particulado, contaminam águas pluviais com o escoamento da água usada no combate, destroem patrimônio e podem se alastrar para áreas verdes adjacentes. A inspeção periódica de equipamentos de combate a incêndio (abrigos de hidrante, mangueiras, registros e prumadas) é uma camada preventiva crítica, regulamentada pelas normas **ABNT NBR 12693** e **ABNT NBR 13714**. A aplicação de visão computacional na verificação visual desses equipamentos barateia auditorias de conformidade e permite varreduras em escala — contribuindo para a redução do risco e do impacto ambiental de incêndios urbanos.

## Aquisição da imagem

| Campo | Valor |
|---|---|
| Origem | Captura presencial pelo aluno (participação ativa) |
| Equipamento | Smartphone |
| Resolução nativa | 1200 × 1600 px (retrato), 3 canais |
| Formato / tamanho | JPEG, ~167 KB |
| Local | Corredor institucional (prédio com placa "Diretoria" visível ao fundo) |
| Iluminação | Artificial interna, branco-amarelada; brilhos especulares no abrigo e na curva da tubulação |
| Sujeito | Abrigo vermelho de hidrante de parede, placa de sinalização "Mangueira de Incêndio", tubulação vertical vermelha e registro de gaveta com volante |

## Estrutura do projeto

```
ProjetoA3-KARLA/
├── processamento.py          # script principal (pipeline completa)
├── requirements.txt          # dependências fixadas
├── README.md                 # este arquivo
├── Artigo_a3.pdf             # relatório técnico (a entregar)
├── imagens/
│   └── imagem_original.jpg   # imagem capturada pelo aluno
└── resultados/
    ├── 01_original.png
    ├── 02_cinza.png
    ├── 03_histograma_rgb.png
    ├── 04_gaussiano.png
    ├── 05_mediana.png
    ├── 06_bilateral.png
    ├── 07_canny.png
    ├── 07b_canny_mascara.png
    ├── 08_otsu.png
    ├── 09_mascara_hsv.png
    ├── 10a_erosao.png
    ├── 10b_dilatacao.png
    ├── 10_morfologia.png
    ├── 11_contornos.png
    ├── 12_grid_final.png
    └── metricas.csv
```

## Requisitos

- Python 3.10 ou superior (testado em 3.13.7)
- Bibliotecas: `opencv-python`, `numpy`, `matplotlib`, `pillow`, `scikit-image`, `pandas`

## Como executar

```bash
# 1) criar ambiente virtual (obrigatório no macOS via Homebrew - PEP 668)
python3 -m venv .venv
source .venv/bin/activate

# 2) instalar dependências
pip install -r requirements.txt

# 3) rodar a pipeline completa
python processamento.py
```

O script lê `imagens/imagem_original.jpg`, executa todas as etapas e grava as figuras + a tabela de métricas em `resultados/`.

## Pipeline implementada

1. **Pré-processamento** — leitura, conversão BGR→RGB, redimensionamento (largura alvo = 600 px), escala de cinza, normalização [0, 1].
2. **Análise da original** — metadados (resolução, canais, tamanho) e histograma dos canais R, G, B.
3. **Filtros** — Gaussiano (σ=1.5), Mediana (5×5) e Bilateral (d=9, σ_color=σ_space=75).
4. **Detecção de bordas** — Canny com limiares 50/150 sobre a imagem cinza suavizada; aplicado também sobre a máscara segmentada para comparação.
5. **Segmentação** — Otsu sobre o canal de cinza **e** segmentação por cor vermelha em HSV (duas faixas em H para cruzar 0°), com H superior limitado a 10 para excluir o laranja dos azulejos.
6. **Morfologia** — erosão, dilatação (separadas) e abertura + fechamento com kernel 5×5.
7. **Contornos** — contornos externos com área mínima de 500 px, sobrepostos em verde sobre a imagem original.
8. **Métricas** — % de área segmentada, valor médio e desvio do pixel (cinza), PSNR e SNR de cada filtro.
9. **Grid final** — 9 painéis comparativos em `12_grid_final.png`.

## Principais resultados obtidos

| Métrica | Valor |
|---|---|
| Pixels totais (após redimensionamento) | 480.000 |
| % de área segmentada (HSV + morfologia) | **23,28%** |
| Valor médio de pixel (cinza) | 156,82 |
| Desvio-padrão (cinza) | 67,51 |
| Limiar Otsu | 131 |
| PSNR / SNR — Gaussiano | 28,09 / 24,68 dB |
| PSNR / SNR — Mediana | 28,29 / 24,88 dB |
| **PSNR / SNR — Bilateral** | **35,47 / 32,07 dB** |
| Contornos detectados (área ≥ 500) | 2 (abrigo + sistema tubo/registro) |

## Observações para a análise crítica

- **Bilateral é claramente o melhor filtro** para esta imagem (PSNR ~7 dB acima dos demais): preserva a borda do abrigo enquanto homogeneíza o vermelho saturado.
- **Otsu falhou**, conforme previsto: o vermelho saturado tem luminância média e acaba agrupado com o quadro escuro da Diretoria — evidência empírica de por que a segmentação por **matiz (HSV)** é mais adequada para esse domínio (invariante à iluminação amarelada).
- **Canny cinza** é dominado pela grade de rejunte dos azulejos; o `07b_canny_mascara.png` mostra que aplicar Canny sobre a máscara HSV elimina esse ruído estrutural.
- A máscara HSV + morfologia capturou exatamente o sistema-alvo: abrigo + tubulação + registro, mais o miolo do símbolo na placa.

## Autoria e disciplina

Projeto desenvolvido para a disciplina de Processamento Digital de Imagens — Atividade A3.
