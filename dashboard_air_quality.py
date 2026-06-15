#  Algoritmo: Isolation Forest  |  Dataset: Air Quality UCI
#  Execute com:  streamlit run dashboard_air_quality.py


import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


st.set_page_config(
    page_title="Air Quality — Detecção de Anomalias",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


COR_NORMAL   = "#4C9BE8"
COR_ANOMALIA = "#E84C4C"
COR_FUNDO    = "#0F1117"
COR_CARD     = "#1E2130"


st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0F1117; }

    /* Cartões de métricas */
    [data-testid="metric-container"] {
        background-color: #1E2130;
        border: 1px solid #2D3147;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: #A0AEC0 !important; font-size: 0.8rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem; font-weight: 700; color: #E2E8F0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13161F; border-right: 1px solid #2D3147; }

    /* Cabeçalho das seções */
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #E2E8F0; border-left: 4px solid #4C9BE8;
        padding-left: 12px; margin: 24px 0 12px 0;
    }

    /* Caixa informativa */
    .info-box {
        background-color: #1E2130; border: 1px solid #2D3147;
        border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
        color: #CBD5E0; line-height: 1.7;
    }

    /* Tag de anomalia */
    .tag-anomalia {
        display: inline-block; background-color: #4A1B1B; color: #FC8181;
        padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .tag-normal {
        display: inline-block; background-color: #1A2E4A; color: #90CDF4;
        padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }

    /* Separador */
    hr { border-color: #2D3147; margin: 32px 0; }

    /* Ocultar menu padrão */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)




# Colunas que o dashboard realmente usa
COLUNAS_UTEIS = ["Date", "Time", "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]

@st.cache_data(show_spinner=False)
def carregar_dataset():
    """Baixa e prepara o dataset Air Quality UCI."""
    import gc
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    df = None
    try:
        import io, zipfile, requests
        resp = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        fname = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        df = pd.read_excel(z.open(fname))
        resp = None  # libera bytes do ZIP da memória
        gc.collect()
    except Exception:
        import os
        locais = ["data/AirQualityUCI.xlsx", "data/AirQualityUCI.csv"]
        for loc in locais:
            if os.path.exists(loc):
                df = pd.read_excel(loc) if loc.endswith(".xlsx") else pd.read_csv(loc, sep=";")
                break
        if df is None:
            st.error("❌ Não foi possível carregar o dataset.")
            st.stop()

    # Descarta colunas desnecessárias (PT08.Sx, NMHC, etc.) o mais cedo possível
    colunas_presentes = [c for c in COLUNAS_UTEIS if c in df.columns]
    df = df[colunas_presentes].copy()

    df = df.dropna(how="all").drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    date_str = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%d/%m/%Y")
    time_str = df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["datetime"] = pd.to_datetime(date_str + " " + time_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # Descarta Date e Time originais após criar datetime
    df = df.drop(columns=["Date", "Time"], errors="ignore")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)

    # Usa categorias para colunas de texto — economiza memória
    df["hour"]      = df["datetime"].dt.hour.astype("int8")
    df["dayofweek"] = df["datetime"].dt.dayofweek.astype("int8")
    df["month"]     = df["datetime"].dt.month.astype("int8")
    df = df.dropna(subset=["datetime"])

    # Converte floats para float32 (metade da memória)
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    gc.collect()
    return df


@st.cache_data(show_spinner=False)
def carregar_modelo():
    import os
    caminhos = ["models/isolation_forest_model.pkl"]
    for c in caminhos:
        if os.path.exists(c):
            with open(c, "rb") as f:
                return pickle.load(f)
    return None


@st.cache_data(show_spinner=False)
def detectar_anomalias(_artefato, _df):
    import gc
    preprocessor = _artefato["preprocessor"]
    model        = _artefato["model"]
    pca_model    = _artefato["pca"]
    feature_cols = _artefato["feature_cols"]

    df_work = _df.copy()
    cols_presentes = [c for c in feature_cols if c in df_work.columns]
    X = preprocessor.transform(df_work[cols_presentes])

    pred = model.predict(X)
    scores = model.decision_function(X)

    df_work["anomalia"]          = np.where(pred == -1, 1, 0).astype("int8")
    df_work["classe"]            = np.where(pred == -1, "Anômalo", "Normal")
    df_work["score_normalidade"] = scores.astype("float32")
    df_work["score_anomalia"]    = (-scores).astype("float32")

    # Libera scores e pred antes do PCA
    del pred, scores
    gc.collect()

    pca_xy = pca_model.transform(X)
    df_work["pca_1"] = pca_xy[:, 0].astype("float32")
    df_work["pca_2"] = pca_xy[:, 1].astype("float32")

    sil = silhouette_score(X, df_work["anomalia"])
    dbi = davies_bouldin_score(X, df_work["anomalia"])
    var_exp = pca_model.explained_variance_ratio_

    # Libera X — maior objeto em memória
    del X, pca_xy
    gc.collect()

    # Descarta coluna score_normalidade (não usada no dashboard)
    df_work = df_work.drop(columns=["score_normalidade"], errors="ignore")

    return df_work, sil, dbi, var_exp, feature_cols



with st.spinner("Carregando dataset e modelo…"):
    df_raw   = carregar_dataset()
    artefato = carregar_modelo()

if artefato is None:
    st.error("❌ Não foi possível carregar o modelo.")
    st.stop()

df_result, silhouette, dbi, var_exp, feature_cols = detectar_anomalias(artefato, df_raw)

# Guarda apenas o que a aba Visão Geral precisa de df_raw, depois libera
_n_registros_originais = len(df_raw)
_n_atributos_numericos = len([c for c in df_raw.select_dtypes(include="number").columns
                               if not c.startswith("Unnamed")])
del df_raw
import gc; gc.collect()


with st.sidebar:
    st.markdown("## 🎛️ Filtros de Análise")
    st.markdown("---")

  
    datas_validas = df_result["datetime"].dropna()
    data_min = datas_validas.min().date()
    data_max = datas_validas.max().date()

    periodo = st.date_input(
        "📅 Período",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
    )

    
    classe_filtro = st.multiselect(
        "🔖 Classificação",
        options=["Normal", "Anômalo"],
        default=["Normal", "Anômalo"],
    )

    
    meses_disp = sorted(df_result["month"].dropna().unique().tolist())
    nomes_meses = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                   7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    meses_sel = st.multiselect(
        "📆 Mês",
        options=meses_disp,
        default=meses_disp,
        format_func=lambda m: nomes_meses.get(m, str(m)),
    )

   
    hora_range = st.slider("⏰ Faixa de Hora", 0, 23, (0, 23))

  
    score_min = float(df_result["score_anomalia"].min())
    score_max = float(df_result["score_anomalia"].max())
    score_filtro = st.slider(
        "📊 Score de Anomalia (mín.)",
        min_value=round(score_min, 3),
        max_value=round(score_max, 3),
        value=round(score_min, 3),
        step=0.001,
    )

    
    cols_sensor = [c for c in ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]
                   if c in df_result.columns]
    col_serie = st.selectbox("📈 Variável — Série Temporal", options=cols_sensor, index=0)

    st.markdown("---")
    st.caption("Air Quality UCI · Isolation Forest")



if len(periodo) == 2:
    d_ini = pd.Timestamp(periodo[0])
    d_fim = pd.Timestamp(periodo[1]) + pd.Timedelta(days=1)
else:
    d_ini = pd.Timestamp(data_min)
    d_fim = pd.Timestamp(data_max) + pd.Timedelta(days=1)

mask = (
    (df_result["datetime"] >= d_ini) &
    (df_result["datetime"] <  d_fim) &
    (df_result["classe"].isin(classe_filtro)) &
    (df_result["month"].isin(meses_sel)) &
    (df_result["hour"].between(hora_range[0], hora_range[1])) &
    (df_result["score_anomalia"] >= score_filtro)
)
df_filtrado = df_result[mask].copy()



st.markdown("""
<div style="text-align:center; padding: 40px 0 20px 0;">
    <div style="font-size:2.8rem;">🌫️</div>
    <h1 style="color:#E2E8F0; font-size:2.2rem; margin:8px 0 4px 0;">
        Detecção de Anomalias — Qualidade do Ar
    </h1>
    <p style="color:#718096; font-size:1rem; margin:0;">
        Isolation Forest &nbsp;·&nbsp; Air Quality UCI Dataset &nbsp;·&nbsp; UCI Machine Learning Repository
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

aba_visao, aba_eda, aba_modelo, aba_anomalias, aba_conclusao = st.tabs([
    "📋 Visão Geral",
    "📊 Análise Exploratória",
    "🤖 Modelo",
    "🚨 Anomalias Detectadas",
    "📝 Conclusão",
])


with aba_visao:

  
    st.markdown('<div class="section-header">Descrição do Projeto</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Este dashboard apresenta os resultados de um sistema de <strong>detecção de anomalias não supervisionada</strong>
    aplicado a registros de qualidade do ar coletados por uma estação multissensorial instalada em uma cidade italiana.
    O objetivo é identificar automaticamente leituras que fogem ao comportamento padrão dos sensores — indicando
    possíveis falhas de equipamento, eventos de poluição extrema ou condições ambientais atípicas.
    <br><br>
    O fluxo do projeto compreende: coleta e limpeza dos dados brutos; análise exploratória; pré-processamento com
    imputação de medianas e padronização; treinamento com <strong>Isolation Forest</strong>; avaliação com métricas
    de separação de grupos; e exportação dos resultados para este dashboard interativo.
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown('<div class="section-header">Informações sobre o Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Nome:</strong> Air Quality UCI Dataset<br>
    <strong>Fonte:</strong> UCI Machine Learning Repository (Vito et al., 2008)<br>
    <strong>Coleta:</strong> Março/2004 a Fevereiro/2005 — frequência horária<br>
    <strong>Sensores:</strong> 5 sensores de óxidos metálicos (PT08.Sx), além de referências analíticas para
    CO, C₆H₆, NOₓ, NO₂ e O₃<br>
    <strong>Variáveis ambientais:</strong> temperatura (T), umidade relativa (RH) e umidade absoluta (AH)<br>
    <strong>Valor sentinela:</strong> −200 representa leituras inválidas/ausentes e foi substituído por NaN<br>
    <strong>Observação:</strong> a coluna NMHC(GT) foi excluída da modelagem por excesso de valores ausentes
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    <div style="background:#1E2130;border:1px solid #2D3147;border-radius:12px;padding:16px;text-align:center;">
        <div style="font-size:2rem;color:#4C9BE8;font-weight:700;">{:,}</div>
        <div style="color:#A0AEC0;font-size:0.85rem;">Registros Originais</div>
    </div>""".format(_n_registros_originais), unsafe_allow_html=True)
    col2.markdown("""
    <div style="background:#1E2130;border:1px solid #2D3147;border-radius:12px;padding:16px;text-align:center;">
        <div style="font-size:2rem;color:#68D391;font-weight:700;">{}</div>
        <div style="color:#A0AEC0;font-size:0.85rem;">Atributos Numéricos</div>
    </div>""".format(_n_atributos_numericos),
    unsafe_allow_html=True)
    col3.markdown("""
    <div style="background:#1E2130;border:1px solid #2D3147;border-radius:12px;padding:16px;text-align:center;">
        <div style="font-size:2rem;color:#F6AD55;font-weight:700;">2004–2005</div>
        <div style="color:#A0AEC0;font-size:0.85rem;">Período de Coleta</div>
    </div>""", unsafe_allow_html=True)

   
    st.markdown('<div class="section-header">Resumo dos Resultados</div>', unsafe_allow_html=True)

    total_reg      = len(df_result)
    qtd_anomalias  = int(df_result["anomalia"].sum())
    perc_anomalias = df_result["anomalia"].mean() * 100
    qtd_filtrado   = len(df_filtrado)
    anom_filtrado  = int(df_filtrado["anomalia"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de Registros",     f"{total_reg:,}")
    c2.metric("Anomalias Detectadas",   f"{qtd_anomalias:,}")
    c3.metric("Taxa de Anomalias",      f"{perc_anomalias:.2f}%")
    c4.metric("Silhouette Score",       f"{silhouette:.4f}")
    c5.metric("Davies-Bouldin Index",   f"{dbi:.4f}")

    st.caption("📌 Os valores acima refletem o modelo completo. Os gráficos e a tabela respondem aos filtros da barra lateral.")

    
    st.markdown('<div class="section-header">Distribuição (Período Filtrado)</div>', unsafe_allow_html=True)
    contagem = df_filtrado["classe"].value_counts().reset_index()
    contagem.columns = ["Classe", "Contagem"]
    fig_pizza = px.pie(
        contagem, names="Classe", values="Contagem",
        color="Classe",
        color_discrete_map={"Normal": COR_NORMAL, "Anômalo": COR_ANOMALIA},
        hole=0.55,
    )
    fig_pizza.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0", legend_font_color="#CBD5E0",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_pizza, use_container_width=True)



with aba_eda:

    st.markdown('<div class="section-header">Distribuição das Variáveis (Histogramas)</div>', unsafe_allow_html=True)

    cols_hist = [c for c in ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]
                 if c in df_filtrado.columns]
    n_cols = 3
    rows = [cols_hist[i:i+n_cols] for i in range(0, len(cols_hist), n_cols)]
    for row in rows:
        colunas_st = st.columns(len(row))
        for j, col_var in enumerate(row):
            dados = df_filtrado[col_var].dropna()
            if dados.empty:
                continue
            fig = px.histogram(
                dados, nbins=40, title=f"Distribuição de {col_var}",
                color_discrete_sequence=[COR_NORMAL],
                labels={col_var: col_var, "count": "Frequência"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E0", margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False,
                title_font_size=13,
            )
            fig.update_xaxes(gridcolor="#2D3147", zerolinecolor="#2D3147")
            fig.update_yaxes(gridcolor="#2D3147", zerolinecolor="#2D3147")
            colunas_st[j].plotly_chart(fig, use_container_width=True)

    
    st.markdown('<div class="section-header">Boxplots por Classificação</div>', unsafe_allow_html=True)

    col_box = st.selectbox("Variável para boxplot", options=cols_hist)
    fig_box = px.box(
        df_filtrado.dropna(subset=[col_box]),
        y=col_box, color="classe",
        color_discrete_map={"Normal": COR_NORMAL, "Anômalo": COR_ANOMALIA},
        points="outliers",
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0", legend_title_text="Classe",
        margin=dict(t=20, b=20),
    )
    fig_box.update_yaxes(gridcolor="#2D3147")
    st.plotly_chart(fig_box, use_container_width=True)

    
    st.markdown('<div class="section-header">Mapa de Correlação</div>', unsafe_allow_html=True)
    num_cols_corr = [c for c in df_result.select_dtypes(include="number").columns
                     if c not in ("anomalia", "hour", "dayofweek", "month", "pca_1", "pca_2",
                                  "score_anomalia") and not c.startswith("Unnamed")]
    corr = df_result[num_cols_corr].corr()

    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0",
        margin=dict(t=20, b=20),
        coloraxis_colorbar_tickfont_color="#CBD5E0",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    
    st.markdown(f'<div class="section-header">Série Temporal — {col_serie}</div>', unsafe_allow_html=True)
    df_serie = df_filtrado.dropna(subset=["datetime", col_serie]).sort_values("datetime")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_serie["datetime"], y=df_serie[col_serie],
        mode="lines", name=col_serie,
        line=dict(color=COR_NORMAL, width=1),
    ))
    anom_serie = df_serie[df_serie["anomalia"] == 1]
    fig_ts.add_trace(go.Scatter(
        x=anom_serie["datetime"], y=anom_serie[col_serie],
        mode="markers", name="Anômalo",
        marker=dict(color=COR_ANOMALIA, size=6, symbol="circle"),
    ))
    fig_ts.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0",
        xaxis=dict(gridcolor="#2D3147", zerolinecolor="#2D3147"),
        yaxis=dict(gridcolor="#2D3147", zerolinecolor="#2D3147"),
        legend=dict(font_color="#CBD5E0"),
        margin=dict(t=20, b=20),
        height=350,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    
    st.markdown('<div class="section-header">Estatísticas Descritivas</div>', unsafe_allow_html=True)
    desc = df_filtrado[cols_hist].describe().T.round(3)
    st.dataframe(desc, use_container_width=True)



with aba_modelo:

    st.markdown('<div class="section-header">Algoritmo Utilizado: Isolation Forest</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Isolation Forest</strong> é um algoritmo de detecção de anomalias baseado em árvores de decisão aleatórias.
    Seu princípio é simples e eficiente: anomalias são pontos raros e diferentes do padrão, e portanto
    são <em>isolados mais rapidamente</em> do que pontos normais quando o espaço de atributos é particionado aleatoriamente.
    <br><br>
    Em cada árvore, o algoritmo seleciona aleatoriamente um atributo e um valor de corte entre o mínimo e o máximo
    daquele atributo. O processo se repete recursivamente. O <strong>path length</strong> (número de divisões necessárias
    para isolar um ponto) é inversamente proporcional ao grau de anomalia: pontos anômalos têm path length curto.
    <br><br>
    <strong>Vantagens:</strong> escala linearmente com o tamanho do dataset; não assume distribuição dos dados;
    eficiente em espaços de alta dimensionalidade; não requer rótulos.<br>
    <strong>Limitação:</strong> o hiperparâmetro <code>contamination</code> define a fração esperada de anomalias
    e deve ser ajustado conforme o domínio do problema.
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown('<div class="section-header">Parâmetros do Modelo</div>', unsafe_allow_html=True)
    mod = artefato["model"]
    params = {
        "n_estimators":  mod.n_estimators,
        "contamination": mod.contamination,
        "max_samples":   mod.max_samples,
        "random_state":  mod.random_state,
        "n_jobs":        mod.n_jobs,
    }
    df_params = pd.DataFrame(list(params.items()), columns=["Parâmetro", "Valor"])
    st.dataframe(df_params, use_container_width=True, hide_index=True)

   
    st.markdown('<div class="section-header">Features Utilizadas</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({"#": range(1, len(feature_cols)+1), "Feature": feature_cols})
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

   
    st.markdown('<div class="section-header">Métricas de Avaliação</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Por ser um método não supervisionado, a avaliação quantitativa usa métricas de separação de grupos:<br><br>
    • <strong>Silhouette Score</strong>: mede a coesão intra-grupo e a separação inter-grupos. Varia de −1 a +1;
      valores próximos de +1 indicam grupos bem definidos.<br>
    • <strong>Davies-Bouldin Index (DBI)</strong>: razão média entre dispersão interna e distância entre grupos.
      Valores menores indicam melhor separação.
    </div>
    """, unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score",      f"{silhouette:.4f}")
    m2.metric("Davies-Bouldin Index",  f"{dbi:.4f}")

    
    st.markdown('<div class="section-header">Visualização PCA — Anomalias × Normais</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    O PCA foi aplicado apenas para visualização, reduzindo os {len(feature_cols)} atributos para 2 componentes
    principais com variância explicada de <strong>{var_exp[0]*100:.1f}%</strong> (PC1) e
    <strong>{var_exp[1]*100:.1f}%</strong> (PC2) — total de <strong>{sum(var_exp)*100:.1f}%</strong>.
    </div>
    """, unsafe_allow_html=True)

    df_pca_plot = df_filtrado.dropna(subset=["pca_1", "pca_2"])
    fig_pca = px.scatter(
        df_pca_plot, x="pca_1", y="pca_2", color="classe",
        color_discrete_map={"Normal": COR_NORMAL, "Anômalo": COR_ANOMALIA},
        opacity=0.6,
        labels={"pca_1": "Componente Principal 1", "pca_2": "Componente Principal 2"},
        hover_data={"score_anomalia": ":.4f"},
    )
    fig_pca.update_traces(marker_size=4)
    fig_pca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0", legend_title_text="Classe",
        xaxis=dict(gridcolor="#2D3147", zerolinecolor="#2D3147"),
        yaxis=dict(gridcolor="#2D3147", zerolinecolor="#2D3147"),
        margin=dict(t=20, b=20), height=450,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    
    st.markdown('<div class="section-header">Distribuição dos Scores de Anomalia</div>', unsafe_allow_html=True)
    fig_score = px.histogram(
        df_filtrado, x="score_anomalia", color="classe", nbins=60,
        barmode="overlay", opacity=0.75,
        color_discrete_map={"Normal": COR_NORMAL, "Anômalo": COR_ANOMALIA},
        labels={"score_anomalia": "Score de Anomalia", "count": "Frequência"},
    )
    fig_score.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0",
        xaxis=dict(gridcolor="#2D3147"), yaxis=dict(gridcolor="#2D3147"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_score, use_container_width=True)



with aba_anomalias:

    df_anom = df_filtrado[df_filtrado["anomalia"] == 1].copy()
    qtd = len(df_anom)
    total_fil = len(df_filtrado)
    perc = qtd / total_fil * 100 if total_fil > 0 else 0

    st.markdown('<div class="section-header">Resumo das Anomalias (Período Filtrado)</div>',
                unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("Total no Período",      f"{total_fil:,}")
    a2.metric("Anomalias no Período",  f"{qtd:,}")
    a3.metric("Taxa no Período",       f"{perc:.2f}%")

    
    st.markdown('<div class="section-header">Anomalias por Mês</div>', unsafe_allow_html=True)
    anom_mes = (
        df_filtrado.groupby("month")["anomalia"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "Anomalias", "count": "Total", "month": "Mês"})
    )
    anom_mes["Taxa (%)"] = (anom_mes["Anomalias"] / anom_mes["Total"] * 100).round(2)
    anom_mes["Mês"] = anom_mes["Mês"].map(
        {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
         7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    )
    fig_mes = px.bar(
        anom_mes, x="Mês", y="Anomalias",
        text="Anomalias", color="Taxa (%)",
        color_continuous_scale="Reds",
    )
    fig_mes.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0",
        xaxis=dict(gridcolor="#2D3147"), yaxis=dict(gridcolor="#2D3147"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_mes, use_container_width=True)

    
    st.markdown('<div class="section-header">Anomalias por Hora do Dia</div>', unsafe_allow_html=True)
    anom_hora = (
        df_filtrado.groupby("hour")["anomalia"]
        .sum().reset_index()
        .rename(columns={"anomalia": "Anomalias", "hour": "Hora"})
    )
    fig_hora = px.bar(
        anom_hora, x="Hora", y="Anomalias",
        color="Anomalias", color_continuous_scale="Reds",
    )
    fig_hora.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E0",
        xaxis=dict(gridcolor="#2D3147", dtick=1),
        yaxis=dict(gridcolor="#2D3147"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_hora, use_container_width=True)

    
    st.markdown('<div class="section-header">Tabela de Registros Anômalos</div>', unsafe_allow_html=True)

    cols_tab = ["datetime", "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
                "T", "RH", "AH", "score_anomalia"]
    cols_tab = [c for c in cols_tab if c in df_anom.columns]

    df_tabela = (
        df_anom[cols_tab]
        .sort_values("score_anomalia", ascending=False)
        .reset_index(drop=True)
    )
    df_tabela["score_anomalia"] = df_tabela["score_anomalia"].round(4)
    df_tabela["datetime"] = df_tabela["datetime"].dt.strftime("%d/%m/%Y %H:%M")

    n_exibir = st.slider("Número de registros a exibir", 10, min(500, len(df_tabela)), 50)
    st.dataframe(df_tabela.head(n_exibir), use_container_width=True)

    
    csv_bytes = df_tabela.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar tabela completa (CSV)",
        data=csv_bytes,
        file_name="anomalias_air_quality.csv",
        mime="text/csv",
    )



with aba_conclusao:

    st.markdown('<div class="section-header">Conclusão</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    O modelo <strong>Isolation Forest</strong> foi aplicado ao dataset Air Quality UCI com
    <strong>{len(df_result):,} registros horários</strong> coletados entre março de 2004 e fevereiro de 2005.
    A taxa de contaminação definida foi de <strong>3%</strong>, resultando na detecção de
    <strong>{qtd_anomalias:,} registros anômalos</strong> ({perc_anomalias:.2f}% do total).
    <br><br>
    As métricas de separação de grupos confirmam a qualidade do modelo:
    o <strong>Silhouette Score de {silhouette:.4f}</strong> indica uma separação moderada entre os grupos,
    enquanto o <strong>Davies-Bouldin Index de {dbi:.4f}</strong> reforça a coesão interna de cada grupo.
    A visualização PCA evidencia que os pontos anômalos, embora distribuídos, tendem a se posicionar
    nas regiões periféricas do espaço de atributos, validando a lógica do algoritmo.
    <br><br>
    A análise temporal revelou que as anomalias não estão concentradas em um único período, o que sugere
    que elas refletem eventos pontuais — como picos de poluição, condições climáticas extremas ou
    falhas transitórias de sensores — em vez de uma deriva sistemática do equipamento.
    <br><br>
    <strong>Limitações e recomendações:</strong>
    <ul>
        <li>O valor de <code>contamination=0.03</code> é uma suposição inicial; uma análise de domínio
            com especialistas poderia calibrar melhor esse parâmetro.</li>
        <li>A ausência de rótulos reais impede a avaliação por métricas supervisionadas (F1, AUC-ROC).
            Uma validação manual de amostra dos registros marcados seria recomendável.</li>
        <li>A extensão do modelo para detecção em tempo real exigiria uma estratégia de atualização
            incremental do pré-processador e revisão periódica do threshold de score.</li>
        <li>A remoção da coluna NMHC(GT) por excesso de valores ausentes pode ter reduzido o poder
            discriminativo do modelo para anomalias relacionadas a hidrocarbonetos não metânicos.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Referências</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    • Vito, S. et al. (2008). <em>On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario</em>.
      Sensors and Actuators B: Chemical.<br>
    • Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). <em>Isolation Forest</em>. ICDM 2008.<br>
    • UCI Machine Learning Repository — Air Quality Dataset:
      <a href="https://archive.ics.uci.edu/ml/datasets/Air+Quality" target="_blank" style="color:#4C9BE8;">
      archive.ics.uci.edu/ml/datasets/Air+Quality</a>
    </div>
    """, unsafe_allow_html=True)