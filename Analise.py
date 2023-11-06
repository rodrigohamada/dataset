import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium import plugins
from folium.plugins import HeatMap
from matplotlib.dates import date2num
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Ler o conjunto de dados com tipos de dados especificados e desabilitando low_memory
dtype = {"LATITUDE": "float64", "LONGITUDE": "float64", "CIDADE": "str"}
df = pd.read_csv(
    "C:\\Users\\rhamada\\Desktop\\Dados\\BO.csv", dtype=dtype, low_memory=False
)

# Remover linhas com valores ausentes nas coordenadas geográficas
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

# Etapa 1: Distribuição Geográfica dos Crimes (Mapa de Calor)
mapa_crimes = folium.Map(
    location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()], zoom_start=10
)
dados_calor = [
    [linha["LATITUDE"], linha["LONGITUDE"]] for indice, linha in df.iterrows()
]
plugins.HeatMap(dados_calor).add_to(mapa_crimes)
mapa_crimes.save("Mapa de Calor.html")

# Gráfico 1: Tipos de Crimes Mais Comuns (Gráfico de Barras)
plt.figure(figsize=(12, 6))
sns.countplot(y="RUBRICA", data=df, order=df["RUBRICA"].value_counts().index[:10])
plt.title("Top 10 Tipos de Crimes Mais Comuns")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Tipo de Crime")
plt.show()

# Modelo de Aprendizado de Máquina para Prever Tipos de Crimes Mais Comuns
# Pré-processamento dos dados
crimes_mais_comuns = df["RUBRICA"].value_counts().index[:10]
df["RUBRICA_NUM"] = df["RUBRICA"].apply(
    lambda x: x if x in crimes_mais_comuns else "Outros"
)

# Dividir os dados em conjuntos de treinamento e teste
X = df[["LATITUDE", "LONGITUDE"]]
y = df["RUBRICA_NUM"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar um modelo de classificação (Random Forest)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_predito = modelo.predict(X_teste)

# Avaliar o desempenho do modelo
acuracia = accuracy_score(y_teste, y_predito)
print(f"Acurácia do modelo - Gráfico 1: {acuracia * 100:.2f}%")

# Visualizar as previsões
dados_previstos = pd.DataFrame(
    {
        "LATITUDE": X_teste["LATITUDE"],
        "LONGITUDE": X_teste["LONGITUDE"],
        "RUBRICA_NUM_PREVISTA": y_predito,
    }
)
print(dados_previstos.head())

# Gráfico 2: Status das Ocorrências (Gráfico de Barras)
contagem_status = df["FLAG_STATUS"].value_counts()
etiquetas_status = contagem_status.index
etiquetas_status = [
    "Concluído" if etiqueta == "C" else "Em Trâmite" if etiqueta == "T" else etiqueta
    for etiqueta in etiquetas_status
]
valores_status = contagem_status.values

# Gráfico de barras para mostrar a distribuição de status
plt.figure(figsize=(10, 6))
plt.bar(etiquetas_status, valores_status)
plt.xlabel("Status da Ocorrência")
plt.ylabel("Número de Ocorrências")
plt.title("Distribuição de Status da Ocorrência")
plt.show()

# Gráfico 3: Mostra a contagem de ocorrências para cada Delegacia.
# Contagem de ocorrências por delegacia
ocorrencias_por_delegacia = df["DELEGACIA"].value_counts()

# Configurando o tamanho do gráfico
plt.figure(figsize=(12, 6))

# Gráfico de barras
ocorrencias_por_delegacia[:10].plot(kind="bar", color="skyblue")
plt.title("Top 10 Delegacias com Mais Ocorrências de Crimes")
plt.xlabel("Delegacia")
plt.ylabel("Número de Ocorrências")
plt.xticks(rotation=45, ha="right")

# Exibir o gráfico
plt.tight_layout()
plt.show()

# IA: Mostra a contagem de ocorrências para cada Delegacia com previsões do modelo Random Forest
df_reduzido = df.sample(frac=0.1, random_state=42)

# Pré-processamento dos dados
contagem_delegacia = df_reduzido["DELEGACIA"].value_counts()
df_reduzido["DELEGACIA_NUM"] = df_reduzido["DELEGACIA"].apply(
    lambda x: x if x in contagem_delegacia else "Outras"
)

# Dividir os dados em conjuntos de treinamento e teste
X = df_reduzido[["LATITUDE", "LONGITUDE"]]
y = df_reduzido["DELEGACIA_NUM"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar e treinar o modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_pred_rf = modelo_rf.predict(X_teste)

# Avaliar o desempenho do modelo de Random Forest
acuracia_rf = accuracy_score(y_teste, y_pred_rf)
print(f"Acurácia do modelo de Random Forest para o Gráfico 3: {acuracia_rf * 100:.2f}%")

# Gerar as previsões para a contagem de ocorrências por delegacia
contagem_delegacia_prevista = pd.DataFrame({"DELEGACIA_NUM_PREDICTED": y_pred_rf})
contagem_delegacia_prevista["DELEGACIA_NUM_PREDICTED"] = contagem_delegacia_prevista[
    "DELEGACIA_NUM_PREDICTED"
].apply(lambda x: x if x in contagem_delegacia else "Outras")

# Contagem de Ocorrências por Delegacia no Conjunto de Dados Reduzido
ocorrencias_por_delegacia = df_reduzido["DELEGACIA_NUM"].value_counts()

# Ordenar o DataFrame
ocorrencias_por_delegacia = ocorrencias_por_delegacia.sort_values(ascending=False)

# Reduzir o número de delegacias exibidas (exemplo: top 10)
top_delegacias = ocorrencias_por_delegacia.index[:10]

# Filtrar as delegacias
df_filtrado = df_reduzido[df_reduzido["DELEGACIA_NUM"].isin(top_delegacias)]

# Criar o gráfico de contagem de ocorrências por delegacia com as previsões do modelo Random Forest
plt.figure(figsize=(12, 6))
sns.barplot(
    x=df_filtrado["DELEGACIA_NUM"].value_counts().values,
    y=df_filtrado["DELEGACIA_NUM"].value_counts().index,
    hue=df_filtrado["DELEGACIA_NUM"].value_counts().index,  # Atribuir 'hue' ao 'y'
    palette="viridis",
    legend=False,  # Remover a legenda
)
plt.title("Previsão de Contagem de Ocorrências por Delegacia (Usando Random Forest)")
plt.xlabel("Número de Ocorrências (Previsto)")
plt.ylabel("Delegacia")
plt.xticks(rotation=45, ha="right")  # Rotacionar as legendas do eixo y
plt.show()

# Gráfico 4: Distribuição de Crimes por Cidade
# Filtro das 10 cidades com os maiores índices de ocorrências
top_10_cidades = df["CIDADE"].value_counts().nlargest(10).index
df_top_10 = df[df["CIDADE"].isin(top_10_cidades)]

# Gráfico de barras
plt.figure(figsize=(12, 6))
sns.countplot(data=df_top_10, y="CIDADE", order=top_10_cidades)
plt.title("Top 10 Cidades com Mais Ocorrências de Crimes")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Cidade")
plt.show()

# Gráfico 5: Análise da distribuição dos diferentes tipos de rubricas (crimes) no conjunto de dados.
# Pré-processamento dos rótulos
rubricas_counts = (
    df["RUBRICA"].str.split("(", n=1).str[0]
)  # Remove "(art.)" e tudo depois do primeiro parêntese
rubricas_counts = rubricas_counts.str.replace(
    "A\.I\.-", "", regex=True
)  # Remove "A.I.-"
rubricas_counts = rubricas_counts.str.replace(
    "\(|\)", "", regex=True
)  # Remove parênteses

# Contagem dos tipos de rubricas (crimes)
rubricas_counts = rubricas_counts.value_counts()

# Plotagem do gráfico de barras com tamanho maior
plt.figure(figsize=(16, 8))
rubricas_counts.plot(kind="bar", color="purple")
plt.title("Distribuição dos Tipos de Rubricas (Crimes)")
plt.xlabel("Tipo de Rubrica")
plt.ylabel("Número de Ocorrências")
plt.xticks(rotation=45, ha="right")  # Rotação e alinhamento dos rótulos no eixo x
plt.tight_layout()  # Ajuste automático de layout para evitar cortes
plt.grid(axis="y")

plt.show()

# Gráfico 6: Balanceamento de Classes
# Filtro dos nomes das rubricas, removendo parênteses e "A.I.-"
df["RUBRICA"] = df["RUBRICA"].str.replace(r"\(.*\)", "", regex=True)
df["RUBRICA"] = df["RUBRICA"].str.replace("A.I.-", "", regex=True)

# Gráfico de barras
plt.figure(figsize=(10, 6))
sns.countplot(x="RUBRICA", data=df, order=df["RUBRICA"].value_counts().index)
plt.title("Balanceamento de Classes (Tipos de Crimes)")
plt.xlabel("Tipo de Crime")
plt.ylabel("Contagem")
plt.xticks(rotation=45, ha="right")  # Rotaciona as legendas para melhor visualização
plt.show()

# Divida os dados em conjuntos de treinamento e teste
X = df[["LATITUDE", "LONGITUDE"]]
y = df["RUBRICA"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Criar e treinar o modelo de Regressão Logística com um número maior de iterações
modelo_logistic = LogisticRegression(max_iter=10000, random_state=42)
modelo_logistic.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_pred_logistic = modelo_logistic.predict(X_teste)

# Avaliar o desempenho do modelo de Regressão Logística
acuracia_logistic = accuracy_score(y_teste, y_pred_logistic)
print(
    f"Acurácia do modelo de Regressão Logística para o Gráfico 6: {acuracia_logistic * 100:.2f}%"
)


# Gráfico 7: Distribuição de Crimes por Cidade
plt.figure(figsize=(12, 6))
sns.countplot(
    y="RUBRICA", data=df, hue="CIDADE", order=df["RUBRICA"].value_counts().index[:10]
)
plt.title("Distribuição de Crimes por Cidade")
plt.xlabel("Número de Ocorrências")
plt.ylabel("Tipo de Crime")
plt.show()

# Gráfico 8: Evolução Temporal de Tipos de Crimes
# Agrupamento dos dados por ano e tipo de crime e conte o número de ocorrências
crimes_contagens = df.groupby(["ANO_BO", "RUBRICA"]).size().unstack()

# Filtrar tipos de crimes únicos para evitar repetições na legenda
tipos_de_crime_exclusivos = df["RUBRICA"].unique()

# Esquema de cores personalizado para os tipos de crimes
colores = sns.color_palette("hsv", len(tipos_de_crime_exclusivos))

# Gráficos de linha para cada tipo de crime com cores diferentes
plt.figure(figsize=(12, 6))
for i, crime_type in enumerate(tipos_de_crime_exclusivos):
    if crime_type in crimes_contagens.columns:
        plt.plot(
            crimes_contagens.index,
            crimes_contagens[crime_type],
            label=crime_type,
            color=colores[i],
        )

plt.title("Evolução Temporal de Tipos de Crimes")
plt.xlabel("Ano")
plt.ylabel("Número de Ocorrências")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
plt.grid(True)
plt.show()

# Filtro para os tipos de crimes
crime_alvo = "Homicídio simples (art. 121)"
df_filtrado = df[df["RUBRICA"] == crime_alvo]

# Agrupamento dos dados por ano e contagem do número de ocorrências
crimes_contagens = (
    df_filtrado.groupby("ANO_BO").size().reset_index(name="NUM_OCORRENCIAS")
)

# Definir o ano como o índice
crimes_contagens.set_index("ANO_BO", inplace=True)

# Modelo de regressão linear
model = LinearRegression()

# Preparação dos dados para treinamento
X = np.array(crimes_contagens.index).reshape(-1, 1)
y = crimes_contagens["NUM_OCORRENCIAS"]

# Treine o modelo
model.fit(X, y)

# Crie um array de anos para fazer previsões
anos_para_prever = np.array([2017, 2018, 2019]).reshape(-1, 1)

# Faça previsões com o modelo
previsoes = model.predict(anos_para_prever)

# Crie um DataFrame com as previsões
previsoes_df = pd.DataFrame(
    {"ANO_BO": anos_para_prever.flatten(), "Previsão": previsoes}
)

# Plote o gráfico das previsões
plt.figure(figsize=(10, 6))
plt.plot(
    crimes_contagens.index,
    crimes_contagens["NUM_OCORRENCIAS"],
    label="Dados Reais",
    marker="o",
)
plt.plot(
    previsoes_df["ANO_BO"],
    previsoes_df["Previsão"],
    label="Previsões",
    linestyle="--",
    marker="o",
)

# Adicione rótulos e legenda
plt.title(f"Previsão de {crime_alvo} ao Longo dos Anos")
plt.xlabel("Ano")
plt.ylabel("Número de Ocorrências")
plt.legend()

# Exiba o gráfico
plt.show()
