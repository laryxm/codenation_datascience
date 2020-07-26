#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures)


# In[53]:


countries = pd.read_csv("countries.csv")


# In[54]:


new_column_names = ["Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"]

countries.columns = new_column_names

countries.head()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[55]:


# Sua análise começa aqui.
countries.head()


# In[56]:


countries.shape


# In[57]:


countries.info()


# In[58]:


countries.isna().sum()


# In[59]:


to_float_list = ["Pop_density", "Coastline_ratio", 
                 "Net_migration","Infant_mortality",
                "Literacy", "Phones_per_1000",
                "Arable", "Crops", "Other", "Climate",
                 "Birthrate", "Deathrate", "Agriculture",
                "Industry", "Service"]

countries[to_float_list] = countries[to_float_list].apply(lambda x: x.str.replace(',','.')).astype(float)

#df = df.stack().str.replace(',','.').unstack()


# In[60]:


countries.describe()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[61]:


def q1():
    countries['Region'] = countries['Region'].str.strip()
    regions = countries['Region'].sort_values().unique()
    return list(regions)
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[62]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    density_bins = discretizer.fit_transform(countries[["Pop_density"]])
    countries['density_bins'] = density_bins.flatten()    
    return countries[countries['density_bins'] == 9]['Country'].nunique()
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[63]:


def q3():
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
    one_hot_encoder.fit_transform(countries[["Region"]]).shape
    return int(one_hot_encoder.fit_transform(countries[["Region", "Climate"]].fillna(countries.mean())).shape[1])
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[64]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[65]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# In[66]:


def q4():
    pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())])
    countries.drop(columns = "density_bins", inplace= True)
    coumns_to_transform = countries.dtypes[(countries.dtypes == "int64")
                |(countries.dtypes == "float64")].index.to_list()
    countries_transformed = countries.copy(deep = True)
    countries_transformed[coumns_to_transform] = pipeline.fit_transform(countries_transformed[coumns_to_transform])
    
    n_arable = pipeline.transform([test_country[2:]])[0][countries.columns.get_loc('Arable')-2]
    return float(n_arable.round(3))
q4()


# In[67]:


countries.head()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[68]:


countries['Net_migration'].describe()


# In[69]:


def q5():
    q1 = countries['Net_migration'].quantile(0.25)
    q3 = countries['Net_migration'].quantile(0.75)
    iqr = q3 - q1
    non_outlier_interval_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]  

    outliers_acima = len(countries[countries['Net_migration'] <non_outlier_interval_iqr[0]]['Net_migration'])
    outliers_abaixo = len(countries[countries['Net_migration'] >non_outlier_interval_iqr[1]]['Net_migration'])
    # Não devemos remover esses outliers, os dados podem ser autênticos
    # e estão em grande quantidade. Removê-los pode trazer prejuízo para a análise
    return (outliers_acima,outliers_abaixo, False)
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[71]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[72]:


def q6():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)

    newsgroups_counts[0]

    words_matrix = pd.DataFrame(newsgroups_counts.toarray(), columns=count_vectorizer.get_feature_names())

    return int(words_matrix["phone"].sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[77]:


def q7():
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    count_vectorizer = CountVectorizer()
    newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(newsgroups_counts)
    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)

    tfidf_matrix = pd.DataFrame(newsgroups_tfidf.toarray(), columns=np.array(count_vectorizer.get_feature_names()))

    tfidf_matrix['phone'].sum()    
    
    return float(tfidf_matrix['phone'].sum().round(3))
q7()


# In[ ]:





# In[ ]:




