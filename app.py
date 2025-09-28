import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Bibliotecas de Modelagem
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from cmdstanpy import CmdStanModel, install_cmdstan # CmdStanPy para Bayes


# Certifique-se de que 'psycopg2-binary' está instalado para a conexão 'sql'

# --- 1. CONFIGURAÇÃO E CONEXÃO COM O BANCO ---

st.set_page_config(
    page_title="Dashboard ENEM: Rede de Ensino (Foco nas Notas)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Conexão: O Streamlit busca as credenciais (secrets) do seu arquivo secrets.toml
conn = st.connection("postgres", type="sql")

# --- 2. CARREGAMENTO DE DADOS COM CACHE ---

@st.cache_data(ttl=3600) # Mantém os dados em cache por 1 hora
def load_data_from_db():
    """Carrega os dados das notas e da rede de ensino do PostgreSQL."""
    st.info("Carregando dados do banco de dados PostgreSQL. Aguarde...")
    try:
        # Consulta SQL SIMPLIFICADA, focando apenas nas notas e na rede de ensino
        sql_query = """SELECT
    nu_ano,
    co_municipio_prova,
    nu_sequencial,
    no_municipio_prova,
    co_uf_prova,
    sg_uf_prova,
    nome_uf_prova,
    municipio_capital_uf_prova,
    regiao_codigo_prova,
    regiao_nome_prova,
    co_escola,
    co_uf_esc,
    sg_uf_esc,
    co_municipio_esc,
    no_municipio_esc,
    tp_localizacao_esc,
    tp_sit_func_esc,
    nota_cn_ciencias_da_natureza AS nota_cn, -- Simplificando nomes longos
    nota_ch_ciencias_humanas AS nota_ch,
    nota_lc_linguagens_e_codigos AS nota_lc,
    nota_mt_matematica AS nota_mt,
    nota_redacao,
    nota_media_5_notas,
    tp_lingua,
    tp_status_redacao,
    tp_dependencia_adm_esc,
    
    -- Coluna de Classificação da Rede de Ensino
    CASE tp_dependencia_adm_esc
       WHEN 'Estadual' THEN 'Pública'
       WHEN 'Federal' THEN 'Pública'
       WHEN 'Privada' THEN 'Privada'
       ELSE 'Não Classificado' 
    END AS Rede_Ensino_Classificada

FROM
    public.ed_enem_2024_resultados

WHERE 
    co_municipio_esc IN (
        '3104502', '3109303', '3109451', '3170404', 
        '5200100', '5300108', '5222203', '5200175', 
        '5200258', '5200308', '5200605', '5200803', 
        '5203203', '5222302', '5204003', '5205307', 
        '5205497', '5205513', '5205802', '5206206', 
        '5207907', '5208004', '5208608', '5213053', 
        '5214606', '5215231', '5215603', '5217302', 
        '5217609', '5219753', '5220009', '5220686', 
        '5221858'
    );
        """
        
        # Conexão e leitura dos dados em um DataFrame
        df = conn.query(sql_query, ttl=600) 

        # --- Limpeza e Feature Engineering Básica ---
        
        df = df.dropna()
        df = df.rename(columns={
            'nota_cn': 'Natureza',
            'nota_ch': 'Humanas',
            'nota_lc': 'Linguagens',
            'nota_mt': 'Matemática',
            'nota_redacao': 'Redação',
            'nota_media_5_notas': 'Media_Geral',
            'rede_ensino_classificada': 'Rede_Ensino',
            'tp_lingua': 'Língua_Estrangeira',
            'tp_localizacao_esc': 'Localização',
            'tp_sit_func_esc': 'Situação_Func'
        })

        df = df.replace({"Estadual": "Pública", "Federal": "Pública", "Privada": "Privada"})
        
        st.success(f"Dados carregados com sucesso! Total de {len(df)} registros.")
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados do banco de dados: {e}.")
        return pd.DataFrame() 

# --- 3. EXECUÇÃO E CONSTRUÇÃO DO DASHBOARD ---

df = load_data_from_db()

if df.empty:
    st.stop() # Interrompe a execução se houver falha no carregamento

# --- Sidebar e Componentes do Dashboard ---

st.title("📊 Dashboard Exploratório do ENEM: Rede de Ensino")
st.markdown("Análise das diferenças de notas entre Escolas **Públicas** e **Privadas**.")

st.sidebar.header("Variáveis de Análise")

# Seleção da disciplina para o Boxplot
disciplinas = ['Media_Geral', 'Matemática', 'Redação', 'Linguagens', 'Natureza', 'Humanas']
disciplina_selecionada = st.sidebar.selectbox(
    "Selecione a Disciplina para Análise:",
    options=disciplinas,
    index=0 # Média Geral
)

# --- SEÇÃO 1: ESTATÍSTICAS DESCRITIVAS DETALHADAS ---
st.header("1. Estatísticas Detalhadas de Notas por Rede de Ensino")
st.markdown("Comparação das principais métricas (Média, Mediana e Desvio Padrão) para cada disciplina.")

def calculate_stats(df):
    """Calcula estatísticas descritivas para cada coluna de nota."""
    stats = df.groupby('Rede_Ensino')[disciplinas].agg(['mean', 'median', 'std']).T
    stats.columns.name = None
    stats.index.names = ['Disciplina', 'Métrica']
    stats = stats.replace({"mean":"Média", "median":"Mediana", "std":"Desvio Padrão"})
    return stats.round(2)

stats_df = calculate_stats(df)
st.dataframe(stats_df, use_container_width=True)

# --- Visualizações ---

# Linha 1: Boxplot da Disciplina Selecionada
st.header(f"2. Distribuição de Notas: {disciplina_selecionada}")

fig_box_rede = px.box(
    df,
    x='Rede_Ensino',
    y=disciplina_selecionada,
    color='Rede_Ensino',
    title=f'Boxplot da Nota de {disciplina_selecionada} por Rede de Ensino',
    points="suspectedoutliers",
    category_orders={"Rede_Ensino": ["Pública", "Privada"]},
    color_discrete_map={'Pública': 'darkorange', 'Privada': 'darkgreen'}
)
fig_box_rede.update_layout(yaxis_title=f"Nota de {disciplina_selecionada}")
st.plotly_chart(fig_box_rede, use_container_width=True)


# Linha 2: Proporção e Contexto

st.header("3. Proporção de Participantes")
colA, colB = st.columns([1, 2])

with colA:
    # Gráfico de Pizza (Distribuição Rede de Ensino)
    rede_counts = df['Rede_Ensino'].value_counts().reset_index()
    rede_counts.columns = ['Rede_Ensino', 'Count']
    
    fig_pie_rede = px.pie(
        rede_counts, 
        values='Count', 
        names='Rede_Ensino', 
        title='Proporção de Participantes por Rede',
        color='Rede_Ensino',
        color_discrete_map={'Pública': 'darkorange', 'Privada': 'darkgreen'}
    )
    fig_pie_rede.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie_rede, use_container_width=True)

with colB:
    st.markdown("""
    <br><br>
    <p style='font-size: 18px;'>
    A grande disparidade na proporção de alunos (maioria da rede <b>Pública</b>) é um conhecimento prévio fundamental. 
    </p>
    <p style='font-size: 18px;'>
    No seu modelo bayesiano, isso deve ser capturado pela prior do intercepto ($\beta_0$), refletindo o log-odds base de ser da rede <b>Privada</b> ($Y=1$), antes de considerarmos as notas individuais.
    </p>
    """, unsafe_allow_html=True)

# Linha 3: Relação Multivariada (Redação vs. Matemática)

st.header("3. Relação entre Notas Chave")

# Scatter plot
# Amostra para gráficos mais leves e rápidos
df_sample = df.sample(frac=0.3, random_state=42) 

fig_scatter = px.scatter(
    df_sample,
    x='Matemática',
    y='Redação',
    color='Rede_Ensino',
    title='Redação vs. Matemática (Amostra)',
    opacity=0.6,
    trendline="ols", # Adiciona uma linha de tendência para visualizar a correlação
    color_discrete_map={'Pública': 'darkorange', 'Privada': 'darkgreen'}
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- SEÇÃO 4: ANÁLISE DO CONTEXTO ESCOLAR ---
st.header("4. Distribuição de Variáveis Contextuais")
st.markdown("Análise da distribuição das características das escolas.")

context_vars = ['Língua_Estrangeira', 'Localização', 'Situação_Func']
col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

for i, col_name in enumerate(context_vars):
    with columns[i]:
        st.subheader(f"Distribuição de {col_name.replace('_', ' ')}")
        
        # Contagem da frequência
        counts = df[col_name].value_counts().reset_index()
        counts.columns = [col_name, 'Count']
        
        # Gráfico de barras
        fig = px.bar(
            counts, 
            x=col_name, 
            y='Count', 
            title=f'Contagem de {col_name.replace("_", " ")}',
            color='Count'
        )
        fig.update_layout(xaxis_title=col_name, yaxis_title="Número de Alunos")
        st.plotly_chart(fig, use_container_width=True)

print(df.columns)
@st.cache_data
def prepare_data_and_run_logit(df):

    X = df[['Matemática', 'Redação', 'Linguagens', 'Natureza', 'Humanas']]
    y = df['Rede_Ensino'] # 1=Privada, 0=Pública
    # Transformação da variável alvo para binária (0=Pública, 1=Privada)
    encoder = OneHotEncoder(drop='if_binary', dtype=int)
    y_encoded = encoder.fit_transform(df[['Rede_Ensino']]).toarray().ravel()
    y = y_encoded
    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_full = scaler.transform(X)

    # --- Modelo Frequentista (Logit) ---
    model_logit = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
    model_logit.fit(X_train_scaled, y_train)

    # Coeficientes
    coefs = pd.DataFrame({
        'Feature': ['Intercept'] + list(X.columns),
        'Coeficiente_Logit': [model_logit.intercept_[0]] + list(model_logit.coef_[0])
    })

    # Métricas
    y_pred = model_logit.predict(X_test_scaled)
    y_proba = model_logit.predict_proba(X_test_scaled)[:, 1]
    
    report = classification_report(y_test, y_pred, target_names=['Pública (0)', 'Privada (1)'], output_dict=True)
    metrics = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Precisão (Privada)': report['Privada (1)']['precision'],
        'Recall (Privada)': report['Privada (1)']['recall']
    }
    
    return X_scaled_full, y, coefs, metrics


# --- 2. MODELO BAYESIANO (CmdStanPy) ---

@st.cache_data
def run_stan_model(X_scaled_full, y_full):
    
    # 2.1. Criação do Modelo Stan (Sintaxe Corrigida)
    stan_code = """
    data {
      int<lower=0> N;          
      int<lower=0> K;          
      matrix[N, K] X;         
      array[N] int<lower=0, upper=1> y; // CORRIGIDO: nova sintaxe de array
    }
    parameters {
      vector[K] beta;           
      real beta0;               
    }
    model {
      // Priors
      beta0 ~ normal(-1, 1.5); 
      beta ~ normal(0.5, 1.0);
      
      // Likelihood
      vector[N] Z = beta0 + X * beta;
      y ~ bernoulli_logit(Z);
    }
    generated quantities {
        // Para prever probabilidades no conjunto completo
        vector[N] p_privada = inv_logit(beta0 + X * beta);
    }
    """
    stan_file_path = 'logit_rede_ensino_cached.stan'
    with open(stan_file_path, 'w') as f:
        f.write(stan_code)

    # 2.2. Preparação de Dados
    stan_data = {
        'N': len(X_scaled_full),
        'K': X_scaled_full.shape[1],
        'X': X_scaled_full,
        'y': y_full
    }

    # 2.3. Compilação e Amostragem
    try:
        model_stan = CmdStanModel(stan_file=stan_file_path)
        fit = model_stan.sample(
            data=stan_data, 
            chains=2, # Menos chains para rodar mais rápido no Streamlit
            iter_warmup=500, 
            iter_sampling=1000, 
            seed=42
        )
        # Retorna o DataFrame com as amostras
        return fit.draws_pd()
    except Exception as e:
        st.error(f"Erro na compilação/amostragem CmdStan: {e}")
        return None


# --- 3. EXECUÇÃO PRINCIPAL DO STREAMLIT ---

st.title("Comparação de Modelos: Previsão Rede de Ensino")


X_scaled_full, y_full, coefs_logit, metrics_logit = prepare_data_and_run_logit(df)

if X_scaled_full is None:
    st.stop()

tab_frequentista, tab_bayesiana = st.tabs(["📊 Resultados Frequentistas (Logit)", "🔬 Resultados Bayesianos (CmdStanPy)"])

# ==============================================================================
# ABA 1: RESULTADOS FREQUENTISTAS
# ==============================================================================
with tab_frequentista:
    st.header("Análise de Coeficientes e Desempenho (Regressão Logística)")

    col_metricas, col_coefs = st.columns(2)

    with col_metricas:
        st.subheader("Métricas de Desempenho")
        st.dataframe(pd.DataFrame(metrics_logit.items(), columns=['Métrica', 'Valor']).set_index('Métrica').round(4), use_container_width=True)
        st.info("A alta AUC-ROC sugere que as notas são excelentes preditores para a Rede de Ensino.")

    with col_coefs:
        st.subheader("Coeficientes (Log-Odds)")
        st.dataframe(coefs_logit.set_index('Feature').round(4), use_container_width=True)
        st.markdown(
            "Os coeficientes positivos e altos indicam que o aumento na nota **padronizada** eleva drasticamente o log-odds de ser uma escola **Privada** ($Y=1$).")
        
        # Gráfico de Coeficientes Frequentistas
        fig_coef_logit = px.bar(
            coefs_logit[coefs_logit['Feature'] != 'Intercept'], 
            x='Feature', 
            y='Coeficiente_Logit', 
            title='Impacto dos Preditores (Log-Odds)',
            color='Feature'
        )
        st.plotly_chart(fig_coef_logit, use_container_width=True)

# ==============================================================================
# ABA 2: RESULTADOS BAYESIANOS
# ==============================================================================
with tab_bayesiana:
    st.header("Distribuições Posteriores e Incerteza (CmdStanPy)")
    st.info("Aguarde a execução do modelo Stan. Isso pode demorar alguns segundos na primeira vez.")

    posterior_samples = run_stan_model(X_scaled_full, y_full)

    if posterior_samples is not None:
        
        # Mapeamento dos nomes do Stan
        betas_map = {
            'beta[1]': 'nota_mt',
            'beta[2]': 'nota_redacao',
            'beta[3]': 'nota_lc',
            'beta0': 'Intercepto'
        }
        
        # Criação do DataFrame de resultados bayesianos
        samples_df = posterior_samples[[*betas_map.keys()]].rename(columns=betas_map)
        
        # Gráfico de Distribuições Posteriores
        st.subheader("Distribuições de Probabilidade (Posteriores)")
        
        fig_dist = go.Figure()
        
        for col in samples_df.columns:
            fig_dist.add_trace(go.Violin(
                y=samples_df[col],
                name=col,
                box_visible=True,
                meanline_visible=True,
                opacity=0.6
            ))
        
        fig_dist.update_layout(
            title="Distribuições Posteriores dos Coeficientes (Log-Odds)",
            yaxis_title="Valor do Coeficiente",
            height=600
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### Resumo da Inferência (Intervalos de Credibilidade)")
        
        # Cálculo do HDI (High Density Interval - 95%)
        def calculate_hdi(series):
            return pd.Series({
                'Média Posterior': series.mean(),
                '2.5% HDI': series.quantile(0.025),
                '97.5% HDI': series.quantile(0.975)
            })

        hdi_df = samples_df.apply(calculate_hdi).T
        st.dataframe(hdi_df.round(4), use_container_width=True)
        
        st.success(
            "O método bayesiano mostra a **distribuição completa** de cada coeficiente (violinos), fornecendo o **Intervalo de Credibilidade (HDI)**. Se o HDI não contiver zero, temos uma alta certeza de que aquele preditor é significativo."
        )