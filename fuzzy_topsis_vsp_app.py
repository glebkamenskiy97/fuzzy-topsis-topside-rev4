import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Fuzzy TOPSIS App", layout="wide")
st.title("Приложение для поддержки принятия решения по выбору рационального варианта верхнего строения морской нефтегазопромысловой платформы")

# --- Ввод данных ---
st.header("Ввод исходных данных")
num_alternatives = st.number_input("Количество альтернатив", min_value=2, value=3)
criteria = ["ЧДД (млн руб.)", "ВНД (%)", "Масса ВСП (т)", "Срок строительства ВСП (мес.)", "Пиковая мощность (МВт)", "Выбросы CO2 (т)"]
benefit_criteria = [True, True, False, False, False, False]

weights_linguistic = {
    "Очень низкая": (0.0, 0.0, 0.1),
    "Низкая": (0.0, 0.1, 0.3),
    "Средняя": (0.2, 0.5, 0.8),
    "Высокая": (0.7, 0.9, 1.0),
    "Очень высокая": (0.9, 1.0, 1.0)
}

st.subheader("Выбор весов критериев (лингвистические оценки)")
weights = []
for crit in criteria:
    sel = st.selectbox(f"Вес критерия '{crit}'", list(weights_linguistic.keys()), index=2)
    weights.append(weights_linguistic[sel])
weights = np.array(weights)

st.subheader("Ввод оценок альтернатив (таблица: l,m,u)")
alt_labels = [f"Альт {i+1}" for i in range(num_alternatives)]
rows = []
for alt in alt_labels:
    for crit in criteria:
        rows.append({"Альтернатива": alt, "Критерий": crit, "Оценка (l,m,u)": "0,0,0"})

data_input = pd.DataFrame(rows)
data_input = st.data_editor(data_input, num_rows="fixed")

# Проверка данных и формирование массива
valid_input = True
data = []
for alt in alt_labels:
    row = []
    for crit in criteria:
        val = data_input[(data_input["Альтернатива"] == alt) & (data_input["Критерий"] == crit)]["Оценка (l,m,u)"].values[0]
        try:
            l, m, u = map(float, val.strip().split(","))
            if not (l < m < u):
                st.warning(f"TFN для {alt}, {crit} должен быть: l < m < u")
                valid_input = False
            row.append((l, m, u))
        except:
            st.error(f"Неверный формат для {alt}, {crit}. Ожидается: l,m,u")
            row.append((0.0, 0.0, 0.0))
            valid_input = False
    data.append(row)
data = np.array(data)

# Функции Fuzzy TOPSIS

def normalize(data, benefit):
    norm = []
    for j in range(data.shape[1]):
        col = data[:, j]
        if benefit[j]:
            max_val = max([u for l, m, u in col])
            norm_col = [(l/max_val, m/max_val, u/max_val) if max_val != 0 else (0, 0, 0) for l, m, u in col]
        else:
            min_val = min([l for l, m, u in col if l > 0]) if any(l > 0 for l, _, _ in col) else 1
            norm_col = [(min_val/u if u != 0 else 0, min_val/m if m != 0 else 0, min_val/l if l != 0 else 0) for l, m, u in col]
        norm.append(norm_col)
    return np.array(norm).T

def weighted_fuzzy_decision(norm_data, weights):
    return np.array([[(r[0]*w[0], r[1]*w[1], r[2]*w[2]) for r, w in zip(row, weights)] for row in norm_data])

def ideal_solutions(weighted_data):
    pis = [(max(weighted_data[:, j, 2]), max(weighted_data[:, j, 1]), max(weighted_data[:, j, 0])) for j in range(weighted_data.shape[1])]
    nis = [(min(weighted_data[:, j, 0]), min(weighted_data[:, j, 1]), min(weighted_data[:, j, 2])) for j in range(weighted_data.shape[1])]
    return pis, nis

def distance(a, b):
    return np.sqrt((1/3)*((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

def closeness(weighted_data, pis, nis):
    cc = []
    for i in range(weighted_data.shape[0]):
        d_pos = np.sqrt(sum([distance(weighted_data[i, j], pis[j])**2 for j in range(weighted_data.shape[1])]))
        d_neg = np.sqrt(sum([distance(weighted_data[i, j], nis[j])**2 for j in range(weighted_data.shape[1])]))
        cc.append(d_neg / (d_pos + d_neg) if (d_pos + d_neg) != 0 else np.nan)
    return cc

# Выбор визуализаций
st.subheader("Выбор способов визуализации результатов")
show_table = st.checkbox("Показать таблицу с итогами", value=True)
show_bar = st.checkbox("Показать столбчатый график", value=True)
show_tfn = st.checkbox("Показать TFN графики по критериям", value=True)
show_radar = st.checkbox("Показать радар-график", value=True)

if st.button("Выполнить расчет и визуализацию") and valid_input:
    norm_data = normalize(data, benefit_criteria)
    weighted_data = weighted_fuzzy_decision(norm_data, weights)
    pis, nis = ideal_solutions(weighted_data)
    cc_scores = closeness(weighted_data, pis, nis)

    results_df = pd.DataFrame({"Альтернатива": alt_labels, "Коэффициент близости": cc_scores})

    if results_df["Коэффициент близости"].isnull().any():
        st.error("Ошибка: один или несколько коэффициентов близости не определены (NaN). Проверьте ввод данных.")
    else:
        results_df["Ранг"] = results_df["Коэффициент близости"].rank(ascending=False).astype(int)
        results_df = results_df.sort_values("Ранг")

        if show_table:
            st.subheader("Таблица результатов")
            st.dataframe(results_df)

        if show_bar:
            st.subheader("Столбчатая диаграмма")
            fig_bar = px.bar(results_df, x="Альтернатива", y="Коэффициент близости", color="Альтернатива")
            st.plotly_chart(fig_bar)

        if show_tfn:
            st.subheader("TFN графики по всем критериям")
            for j, crit in enumerate(criteria):
                fig, ax = plt.subplots()
                for i in range(num_alternatives):
                    tri = data[i, j]
                    ax.plot([tri[0], tri[1], tri[2]], [0, 1, 0], label=f"Альт {i+1}")
                ax.set_title(f"TFN по критерию: {crit}")
                ax.legend()
                st.pyplot(fig)

        if show_radar:
            st.subheader("Радар-график по нормализованным средним значениям")
            radar_df = pd.DataFrame(columns=["Альтернатива"] + criteria)
            for i in range(num_alternatives):
                row = [f"Альтернатива {i+1}"] + [norm_data[i, j][1] for j in range(len(criteria))]
                radar_df.loc[len(radar_df)] = row
            fig_radar = px.line_polar(radar_df, r=criteria, theta=criteria, line_close=True, color=radar_df["Альтернатива"])
            st.plotly_chart(fig_radar)
else:
    if not valid_input:
        st.warning("Пожалуйста, исправьте ошибки в данных перед выполнением расчета.")
