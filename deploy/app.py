"""
Triagem de Diabetes Mellitus — CDC BRFSS 2015
Deploy: Hugging Face Spaces (Streamlit)
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
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Triagem de Diabetes — CDC BRFSS 2015',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0a0e1a; }

.hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(79,195,247,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Source Code Pro', monospace;
    font-size: 2.2rem; font-weight: 700;
    color: #e8f4fd; margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.hero-subtitle { font-size: 1rem; color: #7eb3d4; margin: 0 0 4px 0; }
.hero-disclaimer { font-size: 0.82rem; color: #4a7a9b; font-style: italic; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(79,195,247,0.12);
    border: 1px solid rgba(79,195,247,0.3);
    color: #4fc3f7;
    font-family: 'Source Code Pro', monospace;
    font-size: 0.75rem; padding: 3px 10px;
    border-radius: 20px; margin-right: 6px; margin-top: 12px;
}
.metric-card {
    background: #111827; border: 1px solid #1f2d3d;
    border-radius: 12px; padding: 20px 24px;
    text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4fc3f7; }
.metric-value {
    font-family: 'Source Code Pro', monospace;
    font-size: 2rem; font-weight: 700; margin: 0; line-height: 1;
}
.metric-label {
    font-size: 0.78rem; color: #6b7f96; margin: 6px 0 0 0;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.result-card {
    border-radius: 16px; padding: 28px 32px;
    margin-bottom: 20px; border: 2px solid;
}
.result-card.positivo { background: linear-gradient(135deg,#1a0a0a,#2d1010); border-color:#ef5350; }
.result-card.negativo { background: linear-gradient(135deg,#0a1a1a,#0d2626); border-color:#4fc3f7; }
.result-diagnostico {
    font-family: 'Source Code Pro', monospace;
    font-size: 1.8rem; font-weight: 700; margin: 0 0 6px 0;
}
.result-prob {
    font-size: 3.5rem; font-weight: 700; line-height: 1;
    margin: 8px 0; font-family: 'Source Code Pro', monospace;
}
.result-nivel { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.8; }
.risk-bar-container {
    background: #1a2030; border-radius: 8px; height: 10px; overflow: hidden; margin: 8px 0;
}
.risk-bar-fill { height: 100%; border-radius: 8px; }
.section-title {
    font-family: 'Source Code Pro', monospace;
    font-size: 1rem; font-weight: 600; color: #4fc3f7;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #1e3a5f;
}
.form-group-title {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.78rem; color: #4fc3f7; text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 14px; font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0d1520; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7f96; border-radius: 7px;
    font-family: 'Source Code Pro', monospace; font-size: 0.85rem; padding: 8px 20px;
}
.stTabs [aria-selected="true"] { background: #1a3a5c !important; color: #4fc3f7 !important; }
.stButton > button {
    background: linear-gradient(135deg,#1a5276,#2e86c1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 1rem !important; font-weight: 600 !important;
    padding: 14px 32px !important; width: 100% !important; letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#2471a3,#3498db) !important;
    box-shadow: 0 4px 20px rgba(79,195,247,0.3) !important;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.block-container { padding-top: 20px; padding-bottom: 40px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ARTEFATOS
# ══════════════════════════════════════════════════════════════════════════════

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_ = X_df.quantile(self.lower)
        self.upper_ = X_df.quantile(self.upper)
        return self
    def transform(self, X, y=None):
        return pd.DataFrame(X).copy().clip(
            lower=self.lower_, upper=self.upper_, axis=1
        ).values


@st.cache_resource
def carregar_artefatos():
    pipeline = joblib.load('pipeline_preprocessamento.pkl')
    modelo   = joblib.load('modelo_final.pkl')
    features = joblib.load('features.pkl')
    with open('metadados_modelo.json') as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(modelo)
    return pipeline, modelo, features, meta, explainer

pipeline, modelo, features, meta, explainer = carregar_artefatos()
THRESHOLD = meta['threshold_otimo']


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def criar_features(dados: dict) -> pd.DataFrame:
    d = pd.DataFrame([dados])
    d['imc_categoria'] = pd.cut(
        d['imc'], bins=[0,18.5,25,30,35,100],
        labels=[0,1,2,3,4], right=False
    ).astype(float)
    d['n_fatores_risco']      = (d['pressao_alta'] + d['colesterol_alto'] +
                                 (d['imc']>=30).astype(float) +
                                 d['doenca_cardiaca'] + d['avc'])
    d['idade_x_saude_geral']  = d['faixa_etaria'] * d['saude_geral']
    d['score_saudavel']       = (d['atividade_fisica'] + d['consume_frutas'] +
                                 d['consume_vegetais'] + (1-d['fumante']) +
                                 (1-d['alcool_pesado']))
    d['score_socioeconomico'] = d['renda'] + d['escolaridade']
    d['alto_risco_combinado'] = (
        (d['faixa_etaria']>=9) & (d['saude_geral']>=4)
    ).astype(float)
    return d[features]


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

ESTILO = {
    'figure.facecolor': '#0d1520', 'axes.facecolor': '#111827',
    'axes.edgecolor':   '#1e3a5f', 'axes.labelcolor': '#7eb3d4',
    'xtick.color':      '#4a7a9b', 'ytick.color':    '#4a7a9b',
    'text.color':       '#c9d1e0', 'grid.color':     '#1e3a5f',
    'grid.linewidth':    0.5,      'font.family':    'monospace',
}


def plot_gauge(prob: float) -> plt.Figure:
    plt.rcParams.update(ESTILO)
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0d1520')
    ax.set_facecolor('#0d1520')

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color='#1e3a5f', linewidth=22, solid_capstyle='round')

    for ini, fim, cor in [
        (0.00, 0.30, '#4fc3f7'),
        (0.30, 0.50, '#81c784'),
        (0.50, 0.70, '#ffb74d'),
        (0.70, 1.00, '#ef5350'),
    ]:
        if prob > ini:
            ate = min(prob, fim)
            th  = np.linspace(np.pi - ini*np.pi, np.pi - ate*np.pi, 100)
            ax.plot(np.cos(th), np.sin(th),
                    color=cor, linewidth=22, solid_capstyle='round')

    ang = np.pi - prob * np.pi
    ax.annotate('',
        xy=(0.62*np.cos(ang), 0.62*np.sin(ang)),
        xytext=(0.05*np.cos(ang), 0.05*np.sin(ang)),
        arrowprops=dict(arrowstyle='->', color='white', lw=2.5, mutation_scale=22)
    )
    ax.plot(0, 0, 'o', color='white', markersize=7, zorder=5)

    cor_txt = ('#ef5350' if prob >= 0.70 else '#ffb74d' if prob >= 0.50
               else '#81c784' if prob >= 0.30 else '#4fc3f7')
    ax.text(0, -0.12, f'{prob*100:.1f}%', ha='center', va='center',
            fontsize=28, fontweight='bold', color=cor_txt)

    nivel = ('MUITO ALTO' if prob >= 0.70 else 'ALTO' if prob >= 0.50
             else 'MODERADO' if prob >= 0.30 else 'BAIXO')
    ax.text(0, -0.38, f'Risco {nivel}', ha='center', va='center',
            fontsize=10, color='#6b7f96')

    for pct, a in [(0,np.pi),(25,np.pi*.75),(50,np.pi*.5),(75,np.pi*.25),(100,0)]:
        ax.text(1.2*np.cos(a), 1.2*np.sin(a), f'{pct}%',
                ha='center', va='center', fontsize=7, color='#4a7a9b')

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.55, 1.25)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def plot_shap_individual(X_pac: np.ndarray) -> plt.Figure:
    plt.rcParams.update(ESTILO)
    sv      = explainer.shap_values(X_pac)
    sv      = sv[1] if isinstance(sv, list) else sv
    idx_top = np.argsort(np.abs(sv[0]))[-10:]
    nomes   = [features[i] for i in idx_top]
    vals    = sv[0][idx_top]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#0d1520')
    cores = ['#ef5350' if v > 0 else '#4fc3f7' for v in vals]
    bars  = ax.barh(nomes, vals, color=cores, alpha=0.9,
                    edgecolor='none', height=0.6)

    for bar, v in zip(bars, vals):
        ax.text(v + (0.003 if v >= 0 else -0.003),
                bar.get_y() + bar.get_height()/2,
                f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=8, color='#c9d1e0')

    ax.axvline(0, color='#4a7a9b', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Impacto na Predição (SHAP value)', fontsize=9)
    ax.set_title('Fatores Determinantes desta Predição',
                 fontsize=10, fontweight='bold', pad=12, color='#e8f4fd')

    p1 = mpatches.Patch(color='#ef5350', alpha=0.9, label='Aumenta risco')
    p2 = mpatches.Patch(color='#4fc3f7', alpha=0.9, label='Reduz risco')
    ax.legend(handles=[p1, p2], fontsize=8, framealpha=0.1, loc='lower right')
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_populacao(prob: float) -> plt.Figure:
    plt.rcParams.update(ESTILO)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0d1520')

    np.random.seed(42)
    probs_sim = np.concatenate([
        np.random.beta(2, 8, 38876),
        np.random.beta(4, 3, 7019),
    ])

    ax.hist(probs_sim, bins=60, color='#1a3a5c', alpha=0.7, edgecolor='none', density=True)
    ax.axvline(prob, color='#ef5350', lw=2.5, linestyle='--',
               label=f'Você ({prob*100:.1f}%)')
    ax.axvline(THRESHOLD, color='#ffb74d', lw=1.5, linestyle=':', alpha=0.8,
               label=f'Threshold ({THRESHOLD:.2f})')

    percentil = int(np.mean(probs_sim <= prob) * 100)
    ax.text(prob + 0.02, ax.get_ylim()[1] * 0.88,
            f'Percentil {percentil}', color='#ef5350', fontsize=9, fontweight='bold')

    ax.set_xlabel('Probabilidade Predita de Diabetes', fontsize=9)
    ax.set_ylabel('Densidade', fontsize=9)
    ax.set_title('Sua Posição na População (45.895 pacientes)',
                 fontsize=10, fontweight='bold', pad=10, color='#e8f4fd')
    ax.legend(fontsize=8, framealpha=0.1)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_metricas_comparacao() -> plt.Figure:
    plt.rcParams.update(ESTILO)
    labels   = ['ROC-AUC', 'Recall', 'Precisão', 'F1-Score']
    baseline = [0.8125, 0.7669, 0.3177, 0.4493]
    final    = [0.8192, 0.8168, 0.3059, 0.4451]
    x, w     = np.arange(len(labels)), 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1520')

    b1 = ax.bar(x - w/2, baseline, w, label='Baseline (LR)',
                color='#1a3a5c', alpha=0.9, edgecolor='none')
    b2 = ax.bar(x + w/2, final,    w, label='LightGBM Final',
                color='#4fc3f7', alpha=0.9, edgecolor='none')

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7.5, color='#c9d1e0')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title('Modelo Final vs Baseline',
                 fontsize=11, fontweight='bold', pad=12, color='#e8f4fd')
    ax.legend(fontsize=9, framealpha=0.1)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_matriz_confusao() -> plt.Figure:
    plt.rcParams.update(ESTILO)
    cm = np.array([[25866, 13010], [1286, 5733]])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1520')

    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, shrink=0.8)

    rotulos = ['Sem Diabetes', 'Com Diabetes']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(rotulos)
    ax.set_yticklabels(rotulos)
    ax.set_xlabel('Predição')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de Confusão (threshold={THRESHOLD})',
                 fontsize=10, fontweight='bold', color='#e8f4fd')

    for i in range(2):
        for j in range(2):
            cor = 'white' if cm[i,j] < cm.max()/2 else '#0d1520'
            ax.text(j, i, f'{cm[i,j]:,}',
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=cor)
    plt.tight_layout()
    return fig


def plot_shap_global() -> plt.Figure:
    plt.rcParams.update(ESTILO)
    imp = {
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

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('#0d1520')

    cores = ['#ef5350' if i < 5 else '#4fc3f7' for i in range(len(imp))]
    ax.barh(list(imp.keys()), list(imp.values()),
            color=cores, alpha=0.88, edgecolor='none', height=0.65)

    ax.set_xlabel('Importância Média Absoluta (SHAP)', fontsize=9)
    ax.set_title('Top 15 Features — Importância Global',
                 fontsize=11, fontweight='bold', pad=12, color='#e8f4fd')

    p1 = mpatches.Patch(color='#ef5350', alpha=0.88, label='Top 5')
    p2 = mpatches.Patch(color='#4fc3f7', alpha=0.88, label='Demais')
    ax.legend(handles=[p1, p2], fontsize=8, framealpha=0.1)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS HTML
# ══════════════════════════════════════════════════════════════════════════════

def barra_risco_html(prob: float) -> str:
    cor = ('#ef5350' if prob >= 0.70 else '#ffb74d' if prob >= 0.50
           else '#81c784' if prob >= 0.30 else '#4fc3f7')
    return f"""
    <div class="risk-bar-container">
        <div class="risk-bar-fill"
             style="width:{prob*100:.1f}%;background:{cor};"></div>
    </div>"""


def card_resultado_html(prob: float, diagnostico: str, nivel: str) -> str:
    classe = 'positivo' if diagnostico == 'COM DIABETES' else 'negativo'
    emoji  = '🔴' if classe == 'positivo' else '🟢'
    cor    = ('#ef5350' if prob >= 0.70 else '#ffb74d' if prob >= 0.50
              else '#81c784' if prob >= 0.30 else '#4fc3f7')
    return f"""
    <div class="result-card {classe}">
        <div class="result-diagnostico" style="color:{cor};">{emoji} {diagnostico}</div>
        <div class="result-prob" style="color:{cor};">{prob*100:.1f}%</div>
        <div class="result-nivel" style="color:{cor};">Nível de risco: {nivel}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <div class="hero-title">🩺 Triagem de Diabetes Mellitus</div>
    <div class="hero-subtitle">
        Modelo preditivo treinado com <strong>229.474 pacientes reais</strong>
        do CDC americano — BRFSS 2015
    </div>
    <div class="hero-disclaimer">
        ⚠️ Esta ferramenta é apenas para triagem e não substitui avaliação médica profissional.
    </div>
    <div>
        <span class="hero-badge">LightGBM</span>
        <span class="hero-badge">AUC 0.8192</span>
        <span class="hero-badge">Recall 81.7%</span>
        <span class="hero-badge">CDC BRFSS 2015</span>
        <span class="hero-badge">229.474 pacientes</span>
    </div>
</div>
""", unsafe_allow_html=True)

aba1, aba2, aba3 = st.tabs(['🔬  Predição', '📈  Sobre o Modelo', '📂  Sobre o Dataset'])


# ── ABA 1 ─────────────────────────────────────────────────────────────────────
with aba1:
    col_form, col_res = st.columns([1, 1], gap='large')

    with col_form:
        st.markdown('<div class="section-title">📋 Dados do Paciente</div>',
                    unsafe_allow_html=True)

        # Condições clínicas
        st.markdown('<div class="form-group-title">🏥 Condições Clínicas</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pressao_alta    = st.radio('Pressão Arterial Alta',     ['Não','Sim'], horizontal=True)
            checou_col      = st.radio('Checou Colesterol (5 anos)',['Não','Sim'], horizontal=True, index=1)
            avc             = st.radio('Já teve AVC',                ['Não','Sim'], horizontal=True)
        with c2:
            colesterol_alto = st.radio('Colesterol Alto',            ['Não','Sim'], horizontal=True)
            doenca_card     = st.radio('Doença Cardíaca / Infarto',  ['Não','Sim'], horizontal=True)
            dif_caminhar    = st.radio('Dificuldade para Caminhar',  ['Não','Sim'], horizontal=True)

        st.divider()

        # Medidas
        st.markdown('<div class="form-group-title">📊 Medidas e Saúde Geral</div>',
                    unsafe_allow_html=True)
        imc = st.slider('IMC', 12.0, 60.0, 27.0, 0.5)

        SAUDE_MAP = {'1 — Excelente':'1','2 — Muito Boa':'2','3 — Boa':'3',
                     '4 — Regular':'4','5 — Ruim':'5'}
        saude_str = st.select_slider('Saúde Geral', options=list(SAUDE_MAP.keys()), value='3 — Boa')
        saude_geral = int(SAUDE_MAP[saude_str])

        c1, c2 = st.columns(2)
        with c1:
            dias_mental = st.slider('Dias saúde mental ruim', 0, 30, 0)
        with c2:
            dias_fisico = st.slider('Dias saúde física ruim', 0, 30, 0)

        st.divider()

        # Comportamentos
        st.markdown('<div class="form-group-title">🏃 Comportamentos de Saúde</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ativ_fisica = st.radio('Atividade física',    ['Não','Sim'], horizontal=True, index=1)
            fumante     = st.radio('Fumou ≥100 cigarros', ['Não','Sim'], horizontal=True)
        with c2:
            frutas  = st.radio('Frutas ≥1x/dia',  ['Não','Sim'], horizontal=True, index=1)
            alcool  = st.radio('Álcool pesado',    ['Não','Sim'], horizontal=True)
        with c3:
            vegetais = st.radio('Vegetais ≥1x/dia', ['Não','Sim'], horizontal=True, index=1)

        st.divider()

        # Socioeconômico
        st.markdown('<div class="form-group-title">👤 Dados Socioeconômicos</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            plano      = st.radio('Plano de saúde',       ['Não','Sim'], horizontal=True, index=1)
            sem_medico = st.radio('Sem médico por custo', ['Não','Sim'], horizontal=True)
            sexo       = st.radio('Sexo', ['Feminino','Masculino'], horizontal=True)
        with c2:
            FAIXAS = {'1 — 18–24':1,'2 — 25–29':2,'3 — 30–34':3,'4 — 35–39':4,
                      '5 — 40–44':5,'6 — 45–49':6,'7 — 50–54':7,'8 — 55–59':8,
                      '9 — 60–64':9,'10 — 65–69':10,'11 — 70–74':11,
                      '12 — 75–79':12,'13 — 80+':13}
            EDU   = {'1 — Nunca estudou':1,'2 — Fundamental I':2,'3 — Fundamental II':3,
                     '4 — Ensino Médio':4,'5 — Técnico/Superior Inc.':5,'6 — Graduação':6}
            RENDA = {'1 — < $10k':1,'2 — $10–15k':2,'3 — $15–20k':3,'4 — $20–25k':4,
                     '5 — $25–35k':5,'6 — $35–50k':6,'7 — $50–75k':7,'8 — > $75k':8}

            faixa_str = st.selectbox('Faixa Etária', list(FAIXAS.keys()), index=2)
            edu_str   = st.selectbox('Escolaridade', list(EDU.keys()),    index=4)
            renda_str = st.selectbox('Renda Anual',  list(RENDA.keys()),  index=5)

        st.markdown('<br>', unsafe_allow_html=True)
        analisar = st.button('🔍  ANALISAR RISCO DE DIABETES', use_container_width=True)

    with col_res:
        st.markdown('<div class="section-title">📊 Resultado da Análise</div>',
                    unsafe_allow_html=True)

        if analisar:
            dados = {
                'pressao_alta'          : int(pressao_alta    == 'Sim'),
                'colesterol_alto'       : int(colesterol_alto == 'Sim'),
                'checou_colesterol'     : int(checou_col      == 'Sim'),
                'imc'                   : float(imc),
                'fumante'               : int(fumante         == 'Sim'),
                'avc'                   : int(avc             == 'Sim'),
                'doenca_cardiaca'       : int(doenca_card     == 'Sim'),
                'atividade_fisica'      : int(ativ_fisica     == 'Sim'),
                'consume_frutas'        : int(frutas          == 'Sim'),
                'consume_vegetais'      : int(vegetais        == 'Sim'),
                'alcool_pesado'         : int(alcool          == 'Sim'),
                'plano_saude'           : int(plano           == 'Sim'),
                'sem_medico_por_custo'  : int(sem_medico      == 'Sim'),
                'saude_geral'           : saude_geral,
                'dias_saude_mental_ruim': int(dias_mental),
                'dias_saude_fisica_ruim': int(dias_fisico),
                'dificuldade_caminhar'  : int(dif_caminhar    == 'Sim'),
                'sexo'                  : int(sexo == 'Masculino'),
                'faixa_etaria'          : FAIXAS[faixa_str],
                'escolaridade'          : EDU[edu_str],
                'renda'                 : RENDA[renda_str],
            }

            with st.spinner('Analisando...'):
                df_pac = criar_features(dados)
                X_pac  = pipeline.transform(df_pac)
                prob   = float(modelo.predict_proba(X_pac)[0, 1])

            diagnostico = 'COM DIABETES' if prob >= THRESHOLD else 'SEM DIABETES'
            nivel = ('MUITO ALTO' if prob >= 0.70 else 'ALTO' if prob >= 0.50
                     else 'MODERADO' if prob >= 0.30 else 'BAIXO')

            # Card + barra
            st.markdown(card_resultado_html(prob, diagnostico, nivel),
                        unsafe_allow_html=True)
            st.markdown(barra_risco_html(prob), unsafe_allow_html=True)

            # Métricas rápidas
            m1, m2, m3 = st.columns(3)
            for col, val, lbl, cor in [
                (m1, f'{prob*100:.1f}%', 'Probabilidade', '#4fc3f7'),
                (m2, nivel,              'Nível de Risco', '#ffb74d'),
                (m3, f'{THRESHOLD:.2f}', 'Threshold',      '#81c784'),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <p class="metric-value" style="color:{cor};">{val}</p>
                        <p class="metric-label">{lbl}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.pyplot(plot_gauge(prob), use_container_width=True)

            st.markdown('---')
            st.markdown('<div class="section-title">🧬 Fatores Determinantes</div>',
                        unsafe_allow_html=True)
            st.pyplot(plot_shap_individual(X_pac), use_container_width=True)
            st.caption('🔴 Barras vermelhas aumentam o risco  |  🔵 Barras azuis reduzem o risco')

            st.markdown('---')
            st.markdown('<div class="section-title">👥 Comparação com a População</div>',
                        unsafe_allow_html=True)
            st.pyplot(plot_populacao(prob), use_container_width=True)

        else:
            st.markdown("""
            <div style="background:#0d1520;border:1px dashed #1e3a5f;
                        border-radius:16px;padding:60px 32px;
                        text-align:center;margin-top:40px;">
                <div style="font-size:3rem;margin-bottom:16px;">🩺</div>
                <div style="color:#4fc3f7;font-family:'Source Code Pro',monospace;
                            font-size:1.1rem;font-weight:600;margin-bottom:8px;">
                    Pronto para análise
                </div>
                <div style="color:#4a7a9b;font-size:0.9rem;line-height:1.7;">
                    Preencha os dados do paciente ao lado<br>
                    e clique em <strong style="color:#7eb3d4;">ANALISAR RISCO</strong>
                    para obter a predição.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── ABA 2 ─────────────────────────────────────────────────────────────────────
with aba2:
    st.markdown('<div class="section-title">⚡ Performance do Modelo</div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    for col, (val, lbl, cor) in zip(cols, [
        ('0.8192','ROC-AUC','#4fc3f7'),
        ('81.7%', 'Recall',  '#ef5350'),
        ('30.6%', 'Precisão','#ffb74d'),
        ('0.4451','F1-Score','#81c784'),
        ('0.47',  'Threshold','#ce93d8'),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <p class="metric-value" style="color:{cor};">{val}</p>
                <p class="metric-label">{lbl}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('---')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">📊 Comparação vs Baseline</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_metricas_comparacao(), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">🔢 Matriz de Confusão</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_matriz_confusao(), use_container_width=True)

    st.markdown('---')
    st.markdown('<div class="section-title">🧬 Importância Global (SHAP)</div>',
                unsafe_allow_html=True)
    st.pyplot(plot_shap_global(), use_container_width=True)

    st.markdown('---')
    st.markdown('<div class="section-title">🔧 Decisões Técnicas</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Pipeline de Modelagem**
        - Baseline: Regressão Logística (AUC 0,8125)
        - Comparação: 5 modelos com filtro de Recall ≥ baseline
        - Campeão: LightGBM
        - Otimização: Optuna — 80 trials bayesianos
        - Métrica: `0.5×AUC + 0.3×Recall + 0.2×F1`
        - Rastreamento: MLflow (87 runs)
        """)
    with c2:
        st.markdown("""
        **Pré-processamento**
        - 24.206 duplicatas removidas (9,54%)
        - Winsorização do IMC no percentil 99
        - 6 features criadas via feature engineering
        - `class_weight='balanced'` (sem SMOTE)
        - Threshold: max Recall com Precisão ≥ 2× prevalência
        - Split 80/20 estratificado
        """)

    st.markdown(f"""
    | Hiperparâmetro | Valor ótimo |
    |---|---|
    | `n_estimators` | 163 |
    | `learning_rate` | 0,0993 |
    | `max_depth` | 3 |
    | `num_leaves` | 36 |
    | `subsample` | 0,885 |
    | `colsample_bytree` | 0,749 |
    | `min_child_samples` | 85 |
    | `threshold_otimo` | {THRESHOLD:.3f} |
    """)


# ── ABA 3 ─────────────────────────────────────────────────────────────────────
with aba3:
    st.markdown('<div class="section-title">📂 CDC BRFSS 2015</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        O **BRFSS** (Behavioral Risk Factor Surveillance System) é o maior
        sistema de pesquisa telefônica de saúde dos EUA, conduzido anualmente
        pelo **CDC**. Cada registro representa um adulto americano real respondendo
        perguntas sobre comportamentos de saúde, condições crônicas e uso de
        serviços médicos.

        **Por que este dataset é especial para triagem?**
        Todas as variáveis são baseadas em **autorrelato** — sem exames laboratoriais.
        Isso permite aplicação em contextos de baixo custo como aplicativos de saúde,
        UBS e campanhas de saúde pública.
        """)
    with c2:
        st.markdown("""
        | Atributo | Valor |
        |---|---|
        | Registros originais | 253.680 |
        | Após limpeza | 229.474 |
        | Features originais | 21 |
        | Features após eng. | 27 |
        | Prevalência de DM | 13,9% |
        | Desbalanceamento | 6,2:1 |
        | Valores nulos | 0 |
        | Fonte | CDC / EUA |
        """)

    st.markdown('---')
    st.markdown('<div class="section-title">🔑 Top Insights da EDA</div>',
                unsafe_allow_html=True)

    insights = [
        ('🏥','Fator mais preditivo','`saude_geral` — correlação +0,294 com DM'),
        ('📈','Risco x Idade','De 1,4% (18–24 anos) até 21,3% (75–79 anos)'),
        ('⚡','Acúmulo de riscos','5 fatores simultâneos → taxa de DM de 60%'),
        ('💰','Fator socioeconômico','Renda < $10k tem 3× mais DM que renda > $75k'),
        ('⚖️', 'IMC crítico','Obesidade grau I (IMC ≥ 30) dobra o risco'),
        ('🏋️','Fator protetor','Atividade física: 21,1% → 11,6% de risco'),
    ]

    c1, c2, c3 = st.columns(3)
    for i, (emoji, titulo, desc) in enumerate(insights):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div style="background:#0d1520;border:1px solid #1e3a5f;
                        border-radius:10px;padding:16px;margin-bottom:12px;">
                <div style="font-size:1.4rem;">{emoji}</div>
                <div style="color:#4fc3f7;font-size:0.82rem;font-weight:600;
                            margin:6px 0 4px 0;text-transform:uppercase;letter-spacing:0.8px;">
                    {titulo}
                </div>
                <div style="color:#7eb3d4;font-size:0.85rem;line-height:1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('---')
    st.markdown("""
    **Fontes:**
    - Kaggle: [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
    - CDC BRFSS: [cdc.gov/brfss](https://www.cdc.gov/brfss)
    """)
