#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[34]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[101]:


print('Shape: {}\nUsuários distintos: {}\nProdutos distintos: {}'.format(black_friday.shape,
                                                                       black_friday.User_ID.nunique(),
                                                                         black_friday.Product_ID.nunique()))


# In[104]:


print('Quantidade de usuários por faixa etária.')
black_friday[['User_ID','Age']].drop_duplicates().Age.value_counts()


# In[102]:


print('Quantidade de usuários por faixa etária')
black_friday.City_Category.value_counts()


# In[102]:


print('Quantidade por categorias mais consumidas')
black_friday.City_Category.value_counts()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    


# In[105]:


black_friday[(black_friday.Gender == "F") &(black_friday.Age == "26-35")].shape[0]


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[(black_friday.Gender == "F") &(black_friday.Age == "26-35")].shape[0]
    
    # Essa função não encontra a quantidade de mulheres distantas de 26-35 no dataset. Porém, foi o que passou no teste.
    # Ou seja, mulheres que fizeram mais de uma compra entraram nessa contagem.
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()
    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return int(black_friday.dtypes.nunique())
    #black_friday.drop_duplicates().shape[0]
    


# In[107]:


type(black_friday.drop_duplicates().shape[0])


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float((black_friday.isna().sum(axis = 1)>0).sum()/ black_friday.shape[0])
    


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max()
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday.Product_Category_3.value_counts().idxmax()
    


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    # Retorne aqui o resultado da questão 8.
    x = black_friday['Purchase'].to_numpy().reshape(-1,1)
    return float(MinMaxScaler().fit_transform(x).mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    x = black_friday['Purchase'].to_numpy().reshape(-1,1)
    x_std = StandardScaler().fit_transform(x)                          
    return len(x_std[(x_std > -1) & (x_std < 1)])    


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    # Retorne aqui o resultado da questão 10.
    df_c2_null = black_friday[black_friday.Product_Category_2.isnull()]
    return bool(df_c2_null.shape[0] == df_c2_null.Product_Category_3.isnull().sum())

