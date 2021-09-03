#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# #### Importar a Base de dados

# In[2]:


import pandas as pd
tabela = pd.read_csv("advertising.csv")
display(tabela)


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

sns.pairplot(tabela)
plt.show()

#o pairplot é uma outra representação do heatmap


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[8]:


from sklearn.model_selection import train_test_split

y = tabela["Vendas"]
x= tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#criação das inteligências artificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treinamento das inteligências
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[20]:


from sklearn import metrics

#verificação do resultado dos teste e do desempenho de cada modelo
#criar e previsão do modelo e comparar a previsão do modelo com o resultado real

#criação dos modelos de previsão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#comparação dos modelos
print("resultado percentual regressão linear", metrics.r2_score(y_teste, previsao_regressaolinear))
print("resultado percentual árvore de decisão", metrics.r2_score(y_teste, previsao_arvoredecisao))

#o melhor modelo é o do maior percentual, o que é igual à árvore de decisão


# #### Visualização Gráfica das Previsões

# In[28]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes RegressaoLinear"] = previsao_regressaolinear

plt.figure(figsize=(15,7))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Qual a importância de cada variável para as vendas?

# In[32]:


#gráfico que informa a impotância de cada tipo de anúncio a partir da análise por árvore de decisão

sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()


# #### Será que estamos investindo certo?

# In[ ]:




