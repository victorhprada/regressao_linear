import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados de investimento
dados = {
    'Investimento_Publicidade': [100, 140, 150, 390, 350, 155, 200],
    'Acessos_Site': [200, 260, 300, 760, 700, 320, 400]
}

df = pd.DataFrame(dados)

# Calculando a correlação

print(df['Investimento_Publicidade'].corr(df['Acessos_Site']))

# Passo 1 - Separar a narável dependente da independente

x = df[['Investimento_Publicidade']] # Variável independendte (característica)
y = df['Acessos_Site'] # Variável dependente (rótulo)

# Criando um objeto do tipo LinearRegression(), que será usado para representar nosso modelo de regressão linear.
regressa_acessos_site = LinearRegression()

regressa_acessos_site.fit(x, y)
# O modelo aprende a relação entre as variáveis independetes (x) e a variável dependente (y)
# Após a execução desta linha, o modela estará pronto para fazer previsões

# Atributo do objeto de regressão linear que representa o coeficiente linear da equação
print(regressa_acessos_site.intercept_)

# Atributo do objeto de regressão linear que representa os coeficientes das variáveis independentes na equação
print(regressa_acessos_site.coef_)

# SQR = SOMA DOS QUADRADOS TOTAIS
# SQM = SOMA DOS QUADRADOS DO MODELO
# SQE = SOME DOS QUADRADOS DOS ERROS

# SQT = SQM + SQE
# R2 = SQM / SQT
# R2 = 1 - (SQE/SQT)

print(regressa_acessos_site.score(x, y))

# Plotando os dados originais
plt.scatter(x, y, color='blue', label='Dados Originais')

# Plotar a linha de regressão
plt.plot(x, regressa_acessos_site.predict(x), color='red', label='Linha de Regressão')
plt.legend()
# plt.show()

# Realizando previsões

previsoes = regressa_acessos_site.predict(x)
print(previsoes)
print(y)

investimento = [[500]]
previsao_acessos = regressa_acessos_site.predict(investimento)
print(previsao_acessos)