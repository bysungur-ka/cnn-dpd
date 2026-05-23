import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Ось входной мощности
x = np.linspace(0, 10, 700)

# Параметры линейной области
k = 0.9
b = 0.55

# Рабочая точка: до нее реальная характеристика совпадает с линейной
x_work = 4.0
y_work = k * x_work + b

# Реальная характеристика: до рабочей точки линейная,
# после нее плавный уход в область сжатия усиления
y_real = np.empty_like(x)
mask_linear = x <= x_work
y_real[mask_linear] = k * x[mask_linear] + b

# Нелинейная часть: плавное насыщение после рабочей точки
y_max = 5.45
alpha = 0.62
y_real[~mask_linear] = y_work + (y_max - y_work) * (
    1 - np.exp(-alpha * (x[~mask_linear] - x_work))
)

# Теоретическая / идеальная характеристика: продолжение линейного закона
y_ideal = k * x + b

# Теоретическая точка для того же входа, где реальный отклик уже ниже идеального
x_theory = 6.0
y_theory = k * x_theory + b
y_real_at_theory = np.interp(x_theory, x, y_real)

# Точка P1дБ: условно показываем разность между теоретическим и реальным откликом
# Здесь это не обязательно строго 1 дБ, но подпись показывает смысл.
# При желании можно подобрать x_theory так, чтобы разность была ровно 1 дБ.
diff = y_ideal - y_real
idx_p1db = np.argmin(np.abs(diff - 1.0))
x_p1db = x[idx_p1db]
y_p1db_real = y_real[idx_p1db]
y_p1db_ideal = y_ideal[idx_p1db]

fig, ax = plt.subplots(figsize=(8.4, 5.2))

# Кривые
ax.plot(
    x,
    y_ideal,
    color="#d62728",
    linewidth=2.2,
    label="Теоретическая линейная характеристика",
)
ax.plot(
    x, y_real, color="#1f77b4", linewidth=2.4, label="Реальная характеристика усилителя"
)

# Теоретическая точка и реальный отклик при том же входе
ax.scatter([x_p1db], [y_p1db_ideal], color="#d62728", s=55, zorder=5)
ax.scatter([x_p1db], [y_p1db_real], color="#1f77b4", s=55, zorder=5)

# Направляющие к P1дБ
ax.plot(
    [x_p1db, x_p1db],
    [0, y_p1db_ideal],
    color="gray",
    linestyle="--",
    linewidth=1.0,
    dashes=(5, 4),
)
ax.plot(
    [0, x_p1db],
    [y_p1db_ideal, y_p1db_ideal],
    color="gray",
    linestyle="--",
    linewidth=1.0,
    dashes=(5, 4),
)
ax.plot(
    [0, x_p1db],
    [y_p1db_real, y_p1db_real],
    color="gray",
    linestyle="--",
    linewidth=1.0,
    dashes=(5, 4),
)

# Стрелка 1 дБ
ax.annotate(
    "",
    xy=(x_p1db - 0.65, y_p1db_real),
    xytext=(x_p1db - 0.65, y_p1db_ideal),
    arrowprops=dict(arrowstyle="<->", linewidth=1.0, color="black"),
)
ax.text(
    x_p1db - 0.9,
    (y_p1db_real + y_p1db_ideal) / 2,
    "1 дБ",
    fontsize=10,
    ha="right",
    va="center",
)

# Подписи точек
ax.annotate(
    "Теоретический\nотклик",
    xy=(x_p1db, y_p1db_ideal),
    xytext=(x_p1db - 1.0, y_p1db_ideal + 0.55),
    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"),
    fontsize=10,
    ha="left",
)

ax.annotate(
    "Реальный\nотклик",
    xy=(x_p1db, y_p1db_real),
    xytext=(x_p1db + 1.0, y_p1db_real - 0.55),
    arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"),
    fontsize=10,
    ha="left",
)

ax.text(
    x_p1db + 0.15, y_p1db_real - 0.15, r"$P_{1дБ}$", fontsize=11, ha="left", va="top"
)

# Области
ax.text(1.65, 3.25, "Линейная\nобласть", fontsize=11, ha="center")
ax.text(8.6, 6.2, "Область\nкомпрессии усиления", fontsize=11, ha="center")

# Оси
ax.set_xlabel("Входная мощность, дБм", fontsize=12)
ax.set_ylabel("Выходная мощность, дБм", fontsize=12)

# Схематический рисунок без численных делений
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlim(0, 10)
ax.set_ylim(0, 7.1)
ax.grid(False)
ax.legend(loc="lower right", fontsize=10, frameon=True)

# Оформление как у схемы: без верхней и правой границы
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

fig.tight_layout()

png_path = "pa_gain_compression_linear_until_workpoint_ru.png"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(png_path)
