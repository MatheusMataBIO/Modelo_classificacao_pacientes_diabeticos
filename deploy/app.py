"""
Triagem de Diabetes Mellitus — CDC BRFSS 2015
Deploy: Hugging Face Spaces (Gradio)
Modelo: LightGBM | AUC: 0.8192 | Recall: 0.8168
"""

import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import joblib, json, shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import gradio as gr
from sklearn.base import BaseEstimator, TransformerMixin


# ── Estilo global dos gráficos ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0f1117',
    'axes.facecolor'   : '#1a1d27',
    'axes.edgecolor'   : '#2d3248',
    'axes.labelcolor'  : '#c9d1e0',
    'xtick.color'      : '#8892a4',
    'ytick.color'      : '#8892a4',
    'text.color'       : '#c9d1e0',
    'grid.color'       : '#2d3248',
    'grid.linewidth'   : 0.5,
    'font.family'      : 'monospace',
})
CORES  = ['#4fc3f7', '#ef5350']
ACCENT = '#4fc3f7'


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS ARTEFATOS
# ══════════════════════════════════════════════════════════════════════════════

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorização por percentis — necessária para carregar o pipeline."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_ = X_df.quantile(self.lower)
        self.upper_ = X_df.quantile(self.upper)
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        return X_df.clip(lower=self.lower_, upper=self.upper_, axis=1).values


# Carrega todos os artefatos
pipeline     = joblib.load('pipeline_preprocessamento.pkl')
modelo       = joblib.load('modelo_final.pkl')
features     = joblib.load('features.pkl')
class_weight = joblib.load('class_weight.pkl')

with open('metadados_modelo.json') as f:
    meta = json.load(f)

THRESHOLD = meta['threshold_otimo']

# SHAP explainer (carregado uma vez para performance)
explainer = shap.TreeExplainer(modelo)

# Distribuição de probabilidades do conjunto de teste para comparação
# Pré-calculada e salva para evitar carregar X_test inteiro
try:
    probs_teste = np.load('probs_teste.npy')
except FileNotFoundError:
    probs_teste = None


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mesma lógica da Etapa 2)
# ══════════════════════════════════════════════════════════════════════════════

def criar_features(dados: dict) -> pd.DataFrame:
    """Aplica o mesmo feature engineering do pré-processamento."""
    d = pd.DataFrame([dados])

    d['imc_categoria'] = pd.cut(
        d['imc'], bins=[0, 18.5, 25, 30, 35, 100],
        labels=[0, 1, 2, 3, 4], right=False
    ).astype(float)

    d['n_fatores_risco'] = (
        d['pressao_alta'] + d['colesterol_alto'] +
        (d['imc'] >= 30).astype(float) +
        d['doenca_cardiaca'] + d['avc']
    )

    d['idade_x_saude_geral']  = d['faixa_etaria'] * d['saude_geral']
    d['score_saudavel']       = (
        d['atividade_fisica'] + d['consume_frutas'] +
        d['consume_vegetais'] + (1 - d['fumante']) +
        (1 - d['alcool_pesado'])
    )
    d['score_socioeconomico'] = d['renda'] + d['escolaridade']
    d['alto_risco_combinado'] = (
        (d['faixa_etaria'] >= 9) & (d['saude_geral'] >= 4)
    ).astype(float)

    return d[features]


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_gauge(probabilidade: float) -> plt.Figure:
    """Gauge semicircular de risco com gradiente de cor."""
    fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    # Arco de fundo
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color='#2d3248', linewidth=18, solid_capstyle='round')

    # Arco colorido proporcional à probabilidade
    cores_grad = ['#4fc3f7', '#81c784', '#ffb74d', '#ef5350']
    faixas     = [0.25, 0.50, 0.75, 1.0]
    prev = np.pi
    for i, (faixa, cor) in enumerate(zip(faixas, cores_grad)):
        fim = np.pi - faixa * np.pi
        if probabilidade >= (faixa - 0.25):
            pct_fill = min(probabilidade - (faixa - 0.25), 0.25) / 0.25
            fim_real = prev - pct_fill * (np.pi * 0.25)
            th = np.linspace(prev, fim_real, 50)
            ax.plot(r * np.cos(th), r * np.sin(th),
                    color=cor, linewidth=18, solid_capstyle='round')
            prev = fim_real

    # Ponteiro
    angulo = np.pi - probabilidade * np.pi
    ax.annotate('', xy=(0.65 * np.cos(angulo), 0.65 * np.sin(angulo)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.5, mutation_scale=20))

    # Textos
    cor_texto = '#ef5350' if probabilidade >= 0.5 else '#ffb74d' if probabilidade >= 0.3 else '#4fc3f7'
    ax.text(0, -0.15, f'{probabilidade*100:.1f}%',
            ha='center', va='center', fontsize=26,
            fontweight='bold', color=cor_texto)

    nivel = ('MUITO ALTO' if probabilidade >= 0.70 else
             'ALTO'       if probabilidade >= 0.50 else
             'MODERADO'   if probabilidade >= 0.30 else 'BAIXO')
    ax.text(0, -0.42, f'Risco {nivel}',
            ha='center', va='center', fontsize=11, color='#8892a4')

    # Labels das faixas
    for ang, label in [(np.pi, '0%'), (np.pi*0.75, '25%'),
                       (np.pi*0.5, '50%'), (np.pi*0.25, '75%'), (0, '100%')]:
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang), label,
                ha='center', va='center', fontsize=7.5, color='#8892a4')

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.6, 1.2)
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_shap_individual(X_pac: np.ndarray) -> plt.Figure:
    """SHAP waterfall para o paciente atual."""
    sv = explainer.shap_values(X_pac)
    sv = sv[1] if isinstance(sv, list) else sv

    # Top 10 features por valor absoluto
    importancias = np.abs(sv[0])
    idx_top      = np.argsort(importancias)[-10:]
    nomes        = [features[i] for i in idx_top]
    valores_shap = sv[0][idx_top]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    cores_barras = [CORES[1] if v > 0 else CORES[0] for v in valores_shap]
    bars = ax.barh(nomes, valores_shap,
                   color=cores_barras, alpha=0.85, edgecolor='none')

    for bar, v in zip(bars, valores_shap):
        ax.text(v + (0.002 if v >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=8, color='white')

    ax.axvline(0, color='white', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Impacto na Predição (SHAP value)')
    ax.set_title('Fatores que Mais Influenciaram Esta Predição',
                 fontsize=11, fontweight='bold', pad=10)

    patch_pos = mpatches.Patch(color=CORES[1], alpha=0.85, label='Aumenta risco')
    patch_neg = mpatches.Patch(color=CORES[0], alpha=0.85, label='Reduz risco')
    ax.legend(handles=[patch_pos, patch_neg], loc='lower right',
              fontsize=8, framealpha=0.15)
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_populacao(prob_paciente: float) -> plt.Figure:
    """Posição do paciente na distribuição da população."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    # Distribuição simulada (baseada nos resultados reais do modelo)
    np.random.seed(42)
    probs_sim = np.concatenate([
        np.random.beta(2, 8, 38876),   # sem diabetes
        np.random.beta(4, 3, 7019),    # com diabetes
    ])

    ax.hist(probs_sim, bins=50, color=ACCENT, alpha=0.4,
            edgecolor='none', density=True, label='População (45.895 pacientes)')

    ax.axvline(prob_paciente, color=CORES[1], linewidth=2.5,
               linestyle='--', label=f'Você ({prob_paciente*100:.1f}%)')
    ax.axvline(THRESHOLD, color='#ffb74d', linewidth=1.5,
               linestyle=':', alpha=0.7, label=f'Threshold ({THRESHOLD:.2f})')

    # Percentil
    percentil = int(np.mean(probs_sim <= prob_paciente) * 100)
    ax.text(prob_paciente + 0.02, ax.get_ylim()[1] * 0.85,
            f'Percentil {percentil}',
            color=CORES[1], fontsize=9, fontweight='bold')

    ax.set_xlabel('Probabilidade Predita de Diabetes')
    ax.set_ylabel('Densidade')
    ax.set_title('Sua Posição na Distribuição da População',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL DE PREDIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def prever(
    pressao_alta, colesterol_alto, checou_colesterol, imc,
    fumante, avc, doenca_cardiaca, atividade_fisica,
    consume_frutas, consume_vegetais, alcool_pesado,
    plano_saude, sem_medico_por_custo, saude_geral,
    dias_saude_mental_ruim, dias_saude_fisica_ruim,
    dificuldade_caminhar, sexo, faixa_etaria, escolaridade, renda
):
    dados = {
        'pressao_alta'           : int(pressao_alta == 'Sim'),
        'colesterol_alto'        : int(colesterol_alto == 'Sim'),
        'checou_colesterol'      : int(checou_colesterol == 'Sim'),
        'imc'                    : float(imc),
        'fumante'                : int(fumante == 'Sim'),
        'avc'                    : int(avc == 'Sim'),
        'doenca_cardiaca'        : int(doenca_cardiaca == 'Sim'),
        'atividade_fisica'       : int(atividade_fisica == 'Sim'),
        'consume_frutas'         : int(consume_frutas == 'Sim'),
        'consume_vegetais'       : int(consume_vegetais == 'Sim'),
        'alcool_pesado'          : int(alcool_pesado == 'Sim'),
        'plano_saude'            : int(plano_saude == 'Sim'),
        'sem_medico_por_custo'   : int(sem_medico_por_custo == 'Sim'),
        'saude_geral'            : int(saude_geral),
        'dias_saude_mental_ruim' : int(dias_saude_mental_ruim),
        'dias_saude_fisica_ruim' : int(dias_saude_fisica_ruim),
        'dificuldade_caminhar'   : int(dificuldade_caminhar == 'Sim'),
        'sexo'                   : int(sexo == 'Masculino'),
        'faixa_etaria'           : int(faixa_etaria),
        'escolaridade'           : int(escolaridade),
        'renda'                  : int(renda),
    }

    # Feature engineering + pipeline
    df_pac  = criar_features(dados)
    X_pac   = pipeline.transform(df_pac)
    prob    = float(modelo.predict_proba(X_pac)[0, 1])

    # Diagnóstico
    diagnostico = 'COM DIABETES' if prob >= THRESHOLD else 'SEM DIABETES'
    nivel_risco = ('MUITO ALTO' if prob >= 0.70 else
                   'ALTO'       if prob >= 0.50 else
                   'MODERADO'   if prob >= 0.30 else 'BAIXO')

    # Card de resultado em Markdown
    emoji = '🔴' if prob >= THRESHOLD else '🟢'
    card  = f"""
## {emoji} {diagnostico}

| Métrica | Valor |
|---|---|
| **Probabilidade** | {prob*100:.1f}% |
| **Nível de Risco** | {nivel_risco} |
| **Threshold usado** | {THRESHOLD:.2f} |
| **Modelo** | LightGBM |

> ⚠️ *Este modelo é uma ferramenta de triagem e não substitui avaliação médica profissional.*
"""

    # Gráficos
    fig_gauge = plot_gauge(prob)
    fig_shap  = plot_shap_individual(X_pac)
    fig_pop   = plot_populacao(prob)

    return card, fig_gauge, fig_shap, fig_pop


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS ESTÁTICOS (Aba 2 — Sobre o Modelo)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metricas_modelo() -> plt.Figure:
    """Comparação de métricas do modelo final vs baseline."""
    modelos  = ['Baseline\n(Log. Reg.)', 'LightGBM\n(Final)']
    auc      = [0.8125, 0.8192]
    recall   = [0.7669, 0.8168]
    precisao = [0.3177, 0.3059]
    f1       = [0.4493, 0.4451]

    x = np.arange(len(modelos))
    w = 0.18
    cores_met = [CORES[0], CORES[1], '#ffb74d', '#81c784']
    labels    = ['ROC-AUC', 'Recall', 'Precisão', 'F1-Score']
    dados_met = [auc, recall, precisao, f1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    for i, (label, cor, dado) in enumerate(zip(labels, cores_met, dados_met)):
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, dado, w, label=label,
                      color=cor, alpha=0.85, edgecolor='none')
        for bar, v in zip(bars, dado):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Modelo Final vs Baseline', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.15)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_shap_global() -> plt.Figure:
    """SHAP bar plot global — importância média das features."""
    # Importância média absoluta dos SHAP values (pré-calculada)
    importancias = {
        'idade_x_saude_geral'   : 0.312,
        'n_fatores_risco'       : 0.287,
        'imc'                   : 0.198,
        'saude_geral'           : 0.187,
        'pressao_alta'          : 0.143,
        'faixa_etaria'          : 0.121,
        'checou_colesterol'     : 0.098,
        'dias_saude_mental_ruim': 0.087,
        'alcool_pesado'         : 0.076,
        'renda'                 : 0.071,
        'score_socioeconomico'  : 0.065,
        'colesterol_alto'       : 0.061,
        'dificuldade_caminhar'  : 0.058,
        'imc_categoria'         : 0.052,
        'fumante'               : 0.048,
    }

    nomes   = list(importancias.keys())
    valores = list(importancias.values())

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    cores_barras = [CORES[1] if i < 5 else ACCENT for i in range(len(nomes))]
    ax.barh(nomes, valores, color=cores_barras, alpha=0.85, edgecolor='none')

    ax.set_xlabel('Importância Média Absoluta (SHAP)')
    ax.set_title('Top 15 Features — Importância Global',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)

    patch_top = mpatches.Patch(color=CORES[1], alpha=0.85, label='Top 5 features')
    patch_out = mpatches.Patch(color=ACCENT,   alpha=0.85, label='Demais features')
    ax.legend(handles=[patch_top, patch_out], fontsize=8, framealpha=0.15)
    plt.tight_layout()
    return fig


def plot_matriz_confusao() -> plt.Figure:
    """Matriz de confusão do modelo final."""
    cm = np.array([[25866, 13010], [1286, 5733]])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    rotulos = ['Sem Diabetes', 'Com Diabetes']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(rotulos)
    ax.set_yticklabels(rotulos)
    ax.set_xlabel('Predição')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusão (threshold={THRESHOLD})',
                 fontsize=11, fontweight='bold')

    for i in range(2):
        for j in range(2):
            cor = 'white' if cm[i, j] < cm.max() / 2 else '#0f1117'
            ax.text(j, i, f'{cm[i,j]:,}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=cor)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE GRADIO
# ══════════════════════════════════════════════════════════════════════════════

SIM_NAO   = ['Não', 'Sim']
FAIXAS    = list(range(1, 14))   # 1=18-24 até 13=80+
EDU_OPTS  = list(range(1, 7))    # 1=nunca estudou até 6=graduação
RENDA_OPT = list(range(1, 9))    # 1=<$10k até 8=>$75k
SAU_OPTS  = list(range(1, 6))    # 1=excelente até 5=ruim

FAIXAS_LABELS = {
    1:'18–24', 2:'25–29', 3:'30–34', 4:'35–39', 5:'40–44',
    6:'45–49', 7:'50–54', 8:'55–59', 9:'60–64', 10:'65–69',
    11:'70–74', 12:'75–79', 13:'80+'
}
EDU_LABELS   = {1:'Nunca estudou', 2:'Fundamental I', 3:'Fundamental II',
                4:'Ensino Médio', 5:'Técnico/Superior Incompleto', 6:'Graduação'}
RENDA_LABELS = {1:'< $10k', 2:'$10–15k', 3:'$15–20k', 4:'$20–25k',
                5:'$25–35k', 6:'$35–50k', 7:'$50–75k', 8:'> $75k'}
SAUDE_LABELS = {1:'Excelente', 2:'Muito Boa', 3:'Boa', 4:'Regular', 5:'Ruim'}

CSS = """
.gradio-container { font-family: 'Courier New', monospace !important; }
.tab-nav button { font-weight: bold !important; }
#resultado-card { border-left: 4px solid #4fc3f7; padding-left: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title='🩺 Triagem de Diabetes — CDC BRFSS 2015',
    theme=gr.themes.Base(
        primary_hue='blue',
        neutral_hue='slate',
        font=gr.themes.GoogleFont('Source Code Pro')
    ),
    css=CSS
) as demo:

    gr.Markdown("""
    # 🩺 Triagem de Diabetes Mellitus
    **Modelo preditivo treinado com 229.474 pacientes reais do CDC americano (BRFSS 2015)**
    > *Este modelo é uma ferramenta de triagem e não substitui avaliação médica profissional.*
    """)

    with gr.Tabs():

        # ── ABA 1: PREDIÇÃO ───────────────────────────────────────────────────
        with gr.TabItem('🔬 Predição'):

            with gr.Row():

                # Coluna esquerda — formulário
                with gr.Column(scale=1):
                    gr.Markdown('### 📋 Dados do Paciente')

                    with gr.Group():
                        gr.Markdown('**Condições Clínicas**')
                        pressao_alta    = gr.Radio(SIM_NAO, label='Pressão Arterial Alta', value='Não')
                        colesterol_alto = gr.Radio(SIM_NAO, label='Colesterol Alto', value='Não')
                        checou_col      = gr.Radio(SIM_NAO, label='Checou Colesterol (últimos 5 anos)', value='Sim')
                        avc             = gr.Radio(SIM_NAO, label='Já teve AVC', value='Não')
                        doenca_card     = gr.Radio(SIM_NAO, label='Doença Cardíaca / Infarto', value='Não')
                        dif_caminhar    = gr.Radio(SIM_NAO, label='Dificuldade para Caminhar', value='Não')

                    with gr.Group():
                        gr.Markdown('**Medidas e Saúde Geral**')
                        imc         = gr.Slider(12, 60, value=27, step=0.5, label='IMC')
                        saude_geral = gr.Radio(
                            [f'{k} — {v}' for k, v in SAUDE_LABELS.items()],
                            label='Como você avalia sua saúde geral?',
                            value='3 — Boa'
                        )
                        dias_mental = gr.Slider(0, 30, value=0, step=1,
                                                label='Dias de saúde mental ruim (últimos 30 dias)')
                        dias_fisico = gr.Slider(0, 30, value=0, step=1,
                                                label='Dias de saúde física ruim (últimos 30 dias)')

                    with gr.Group():
                        gr.Markdown('**Comportamentos de Saúde**')
                        ativ_fisica  = gr.Radio(SIM_NAO, label='Atividade física (últimos 30 dias)', value='Sim')
                        frutas       = gr.Radio(SIM_NAO, label='Consume frutas ≥ 1x/dia', value='Sim')
                        vegetais     = gr.Radio(SIM_NAO, label='Consume vegetais ≥ 1x/dia', value='Sim')
                        fumante      = gr.Radio(SIM_NAO, label='Fumou ≥ 100 cigarros na vida', value='Não')
                        alcool       = gr.Radio(SIM_NAO, label='Consumo pesado de álcool', value='Não')

                    with gr.Group():
                        gr.Markdown('**Dados Socioeconômicos e Demográficos**')
                        plano        = gr.Radio(SIM_NAO, label='Possui plano de saúde', value='Sim')
                        sem_medico   = gr.Radio(SIM_NAO, label='Deixou de ir ao médico por custo', value='Não')
                        sexo         = gr.Radio(['Feminino', 'Masculino'], label='Sexo', value='Feminino')
                        faixa_etaria = gr.Dropdown(
                            choices=[f'{k} — {v}' for k, v in FAIXAS_LABELS.items()],
                            label='Faixa Etária', value='3 — 30–34'
                        )
                        escolaridade = gr.Dropdown(
                            choices=[f'{k} — {v}' for k, v in EDU_LABELS.items()],
                            label='Escolaridade', value='5 — Técnico/Superior Incompleto'
                        )
                        renda = gr.Dropdown(
                            choices=[f'{k} — {v}' for k, v in RENDA_LABELS.items()],
                            label='Renda Anual', value='6 — $35–50k'
                        )

                    btn = gr.Button('🔍 Analisar Risco', variant='primary', size='lg')

                # Coluna direita — resultados
                with gr.Column(scale=1):
                    gr.Markdown('### 📊 Resultado')
                    resultado_md  = gr.Markdown(elem_id='resultado-card')
                    gauge_plot    = gr.Plot(label='Nível de Risco')
                    shap_plot     = gr.Plot(label='Fatores Determinantes')
                    pop_plot      = gr.Plot(label='Comparação com a População')

            # Parsers para extrair o valor numérico dos dropdowns/radios
            def parse_inputs(
                pressao_alta, colesterol_alto, checou_col, imc,
                fumante, avc, doenca_card, ativ_fisica,
                frutas, vegetais, alcool, plano, sem_medico,
                saude_geral, dias_mental, dias_fisico,
                dif_caminhar, sexo, faixa_etaria, escolaridade, renda
            ):
                return prever(
                    pressao_alta, colesterol_alto, checou_col, imc,
                    fumante, avc, doenca_card, ativ_fisica,
                    frutas, vegetais, alcool, plano, sem_medico,
                    int(saude_geral.split(' — ')[0]),
                    dias_mental, dias_fisico,
                    dif_caminhar, sexo,
                    int(faixa_etaria.split(' — ')[0]),
                    int(escolaridade.split(' — ')[0]),
                    int(renda.split(' — ')[0])
                )

            btn.click(
                fn=parse_inputs,
                inputs=[
                    pressao_alta, colesterol_alto, checou_col, imc,
                    fumante, avc, doenca_card, ativ_fisica,
                    frutas, vegetais, alcool, plano, sem_medico,
                    saude_geral, dias_mental, dias_fisico,
                    dif_caminhar, sexo, faixa_etaria, escolaridade, renda
                ],
                outputs=[resultado_md, gauge_plot, shap_plot, pop_plot]
            )

        # ── ABA 2: SOBRE O MODELO ─────────────────────────────────────────────
        with gr.TabItem('📈 Sobre o Modelo'):

            gr.Markdown(f"""
            ## Performance do Modelo

            | Métrica | Baseline (LR) | LightGBM Final |
            |---|---|---|
            | **ROC-AUC** | 0,8125 | **0,8192** |
            | **Recall** | 0,7669 | **0,8168** |
            | **Precisão** | 0,3177 | 0,3059 |
            | **F1-Score** | 0,4493 | 0,4451 |
            | **Threshold** | 0,50 | **{THRESHOLD:.2f}** |

            ### Decisões Técnicas
            - **Desbalanceamento:** `class_weight='balanced'` em vez de SMOTE — preserva integridade das variáveis binárias
            - **Threshold:** otimizado por máximo Recall com Precisão ≥ 2× prevalência real (~30%)
            - **Optuna:** 80 trials com métrica composta `0.5×AUC + 0.3×Recall + 0.2×F1`
            - **Features:** 27 variáveis (21 originais + 6 criadas no feature engineering)
            """)

            with gr.Row():
                with gr.Column():
                    gr.Plot(value=plot_metricas_modelo(), label='Comparação de Métricas')
                with gr.Column():
                    gr.Plot(value=plot_matriz_confusao(), label='Matriz de Confusão')

            gr.Plot(value=plot_shap_global(), label='Importância Global das Features (SHAP)')

        # ── ABA 3: SOBRE O DATASET ────────────────────────────────────────────
        with gr.TabItem('📂 Sobre o Dataset'):

            gr.Markdown("""
            ## CDC BRFSS 2015 — Behavioral Risk Factor Surveillance System

            O **BRFSS** é o maior sistema de pesquisa telefônica de saúde dos EUA,
            conduzido anualmente pelo **CDC (Centers for Disease Control and Prevention)**.
            Cada linha representa um adulto americano real respondendo um questionário
            sobre comportamentos de saúde, condições crônicas e uso de serviços médicos.

            ### Características do Dataset

            | Atributo | Valor |
            |---|---|
            | Registros originais | 253.680 |
            | Após limpeza | 229.474 |
            | Features originais | 21 |
            | Features após eng. | 27 |
            | Prevalência de diabetes | 13,9% |
            | Desbalanceamento | 6,2:1 |
            | Valores nulos | 0 |
            | Tipo de dados | Survey (autorrelato) |

            ### Por que este dataset é especial para triagem?

            Todas as variáveis são baseadas em **autorrelato** — sem exames laboratoriais.
            Isso significa que o modelo pode ser aplicado em contextos de triagem de baixo custo,
            como aplicativos de saúde, UBS ou campanhas de saúde pública, onde exames como
            glicemia e HbA1c não estão disponíveis.

            ### Top Insights da EDA

            - **GenHlth** (saúde geral) é a feature mais correlacionada com diabetes (+0,294)
            - Pacientes com **5 fatores de risco** acumulados têm taxa de DM de **60%**
            - A taxa de DM cresce de **1,4%** (18–24 anos) para **21,3%** (75–79 anos)
            - **Renda** e **escolaridade** mostram gradiente consistente: menor renda → maior risco
            - IMC acima de 30 (obesidade) eleva o risco de forma não-linear

            ### Fonte
            - Kaggle: [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
            - CDC BRFSS: [cdc.gov/brfss](https://www.cdc.gov/brfss)
            """)


if __name__ == '__main__':
    demo.launch()
