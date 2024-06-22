import matplotlib.pyplot as plt

# Dados para o segundo gráfico
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
x2 = [1, 2, 3, 4, 5]
y2 = [5, 4, 3, 2, 1]

# Primeiro gráfico
plt.figure(figsize=(12, 6))

# Subplot para o primeiro gráfico
plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, primeiro subplot
plt.plot(x, y, marker='o')
plt.title("Gráfico 1")
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")

# Texto para o primeiro gráfico
min_y = min(y)
text_y = min_y - 1
min_x = min(x)
text_x = min_x + 2
plt.figtext(text_x, text_y, "Texto para o gráfico 1", wrap=True, horizontalalignment='center', fontsize=12)

# Segundo gráfico
# Subplot para o segundo gráfico
plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, segundo subplot
plt.plot(x2, y2, marker='o')
plt.title("Gráfico 2")
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")

# Texto para o segundo gráfico
min_y2 = min(y2)
text_y2 = min_y2 - 1
min_x = min(x)
text_x = min_x + 2
plt.figtext(text_x, text_y2, "Texto para o gráfico 2", wrap=True, horizontalalignment='center', fontsize=12)

plt.tight_layout()  # Ajuste automático dos subplots para evitar sobreposição

plt.show()
