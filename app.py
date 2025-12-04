# app_dark.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SPK - SAW & WP (Dark UI)", layout="wide", initial_sidebar_state="expanded")

# ---------- CUSTOM DARK CSS ----------
dark_css = """
<style>
/* Page background */
body, .stApp {
  background: #0f1720;
  color: #e6eef6;
}

/* Card */
.card {
  background: linear-gradient(180deg, rgba(22,27,34,0.9), rgba(12,16,22,0.85));
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
  border: 1px solid rgba(255,255,255,0.03);
  margin-bottom: 12px;
}

/* Headings */
h1, h2, h3, h4 {
  color: #e6eef6;
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#0ea5a4,#06b6d4);
  color: #021018;
  border: none;
  padding: 8px 14px;
  border-radius: 8px;
  font-weight: 600;
  box-shadow: 0 6px 14px rgba(6,182,212,0.12);
}

/* Inputs */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div>select {
  background: #0b1220;
  color: #e6eef6;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.04);
  padding: 8px;
}

/* Table - embed minimal style */
.table-dark {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.table-dark th, .table-dark td {
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.table-dark th {
  text-align: left;
  color: #cde7f0;
  font-weight: 700;
}
.table-dark td {
  color: #d7eaf6;
}

/* highlight best */
.best {
  background: linear-gradient(90deg, rgba(6,182,212,0.08), rgba(16,185,129,0.06));
  color: #bff0ea;
  border-left: 4px solid #06b6d4;
  border-radius: 6px;
}
.small-muted { color: #9fb6c7; font-size: 13px; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ---------- DEFAULT CONFIG ----------
default_weights = {
    "C1 (Harga)": 0.05,
    "C2 (Kemahiran)": 0.25,
    "C3 (Komputer Tua)": 0.15,
    "C4 (Desktop/Server)": 0.15,
    "C5 (Kesulitan Instalasi)": 0.25,
    "C6 (Stabilitas)": 0.15
}

criteria_type = {
    "C1 (Harga)": "benefit",
    "C2 (Kemahiran)": "cost",
    "C3 (Komputer Tua)": "benefit",
    "C4 (Desktop/Server)": "benefit",
    "C5 (Kesulitan Instalasi)": "benefit",
    "C6 (Stabilitas)": "benefit"
}

criteria_dropdown = {
    "C1 (Harga)": {"Gratis": 4, "Berbayar": 1},
    "C2 (Kemahiran)": {"Tidak Mahir": 4, "Kurang Mahir": 5, "Sedang": 7, "Mahir": 9},
    "C3 (Komputer Tua)": {"Tidak": 5, "Ya": 10},
    "C4 (Desktop/Server)": {"Desktop": 10, "Server": 5},
    "C5 (Kesulitan Instalasi)": {"Grafis": 15, "Command Line": 10},
    "C6 (Stabilitas)": {"Stabil": 9, "Bleeding Edge": 6},
}

criteria_labels = list(default_weights.keys())

# ---------- SESSION STATE ----------
if "weights" not in st.session_state:
    st.session_state.weights = default_weights.copy()

if "alternatives" not in st.session_state:
    st.session_state.alternatives = {
        "Ubuntu": ["Gratis", "Kurang Mahir", "Tidak", "Desktop", "Grafis", "Stabil"],
        "Fedora": ["Gratis", "Sedang", "Tidak", "Desktop", "Grafis", "Stabil"],
        "Arch Linux": ["Gratis", "Mahir", "Ya", "Desktop", "Command Line", "Bleeding Edge"],
        "Linux Mint": ["Gratis", "Tidak Mahir", "Ya", "Desktop", "Grafis", "Stabil"],
        "Kali Linux": ["Gratis", "Mahir", "Tidak", "Desktop", "Grafis", "Bleeding Edge"],
    }

# ---------- HELPERS: SAW & WP ----------
def calculate_saw(values, weights):
    arr = np.array(values, dtype=float)
    norm = np.zeros_like(arr)
    for j, c in enumerate(criteria_labels):
        if criteria_type[c] == "benefit":
            maxv = arr[:, j].max()
            norm[:, j] = arr[:, j] / (maxv if maxv != 0 else 1)
        else:
            minv = arr[:, j].min()
            norm[:, j] = minv / (arr[:, j] + 0.0)
    scores = norm.dot(np.array(list(weights.values())))
    return scores, norm

def calculate_wp(values, weights):
    arr = np.array(values, dtype=float)
    S = []
    w_list = list(weights.values())
    for row in arr:
        ln_s = 0.0
        for j, c in enumerate(criteria_labels):
            if criteria_type[c] == "cost":
                ln_s += (-w_list[j]) * math.log(row[j])
            else:
                ln_s += (w_list[j]) * math.log(row[j])
        S.append(math.exp(ln_s))
    S = np.array(S)
    V = S / S.sum() if S.sum() != 0 else S
    return S, V

# ---------- LAYOUT: SIDEBAR MENU ----------
st.sidebar.markdown("<div class='card'><h3>☁️ SPK - SAW & WP</h3><div class='small-muted'>Pilih menu di bawah</div></div>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["Dashboard", "Pengaturan Bobot", "Data & CRUD", "Hasil Perhitungan"])

# ---------- HEADER ----------
st.markdown("<div class='card'><h1>☁️ Sistem Pendukung Keputusan — SAW & WP</h1><div class='small-muted'>Analisis Komparasi Metode Fuzzy SAW dan Fuzzy Weighted Product (WP) dalam Sistem Pendukung Keputusan Pemilihan Distro Linux untuk Mahasiswa IT</div></div>", unsafe_allow_html=True)

# ---------- MENU: DASHBOARD ----------
if menu == "Dashboard":
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("<div class='card'><h3>Overview</h3><div class='small-muted'>Jumlah alternatif dan ringkasan bobot saat ini</div>", unsafe_allow_html=True)
        st.write("")
        df_alt = pd.DataFrame.from_dict(st.session_state.alternatives, orient='index', columns=criteria_labels)
        st.dataframe(df_alt.style.set_properties(**{"background-color": "#0b1220", "color":"#dff7fb"}))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Bobot Saat Ini</h3>", unsafe_allow_html=True)
        for k, v in st.session_state.weights.items():
            st.markdown(f"<div style='display:flex;justify-content:space-between'><div class='small-muted'>{k}</div><div><b>{v:.3f}</b></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    """, unsafe_allow_html=True)

# ---------- MENU: PENGATURAN BOBOT ----------
elif menu == "Pengaturan Bobot":
    st.markdown("<div class='card'><h3>⚙️ Pengaturan Bobot Kriteria</h3><div class='small-muted'>Atur bobot (pastikan total = 1.0)</div></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    temp = {}
    i=0
    for k in st.session_state.weights.keys():
        with cols[i%2]:
            val = st.number_input(k, min_value=0.0, max_value=1.0, value=float(st.session_state.weights[k]), step=0.01, format="%.2f", key=f"w_{k}")
            temp[k] = val
        i+=1
    # apply
    if st.button("Simpan Bobot"):
        s = sum(temp.values())
        if abs(s - 1.0) > 1e-6:
            st.error(f"Total bobot sekarang = {s:.4f}. Harap pastikan jumlah bobot = 1.0")
        else:
            st.session_state.weights = temp
            st.success("✅ Bobot tersimpan.")

# ---------- MENU: DATA & CRUD ----------
elif menu == "Data & CRUD":
    st.markdown("<div class='card'><h3>🗂️ Data Alternatif (CRUD)</h3><div class='small-muted'>Tambah, edit, atau hapus alternatif</div></div>", unsafe_allow_html=True)

    # show existing
    st.subheader("Daftar Alternatif")
    names = list(st.session_state.alternatives.keys())
    df_show = pd.DataFrame.from_dict(st.session_state.alternatives, orient='index', columns=criteria_labels)
    df_show.index.name = "Alternatif"
    st.table(df_show.style.set_properties(**{"background-color":"#0b1220","color":"#d7eef6"}))

    st.markdown("---")
    st.subheader("✚ Tambah Alternatif Baru")
    new_name = st.text_input("Nama Alternatif", value="")
    new_vals = []
    cols = st.columns(3)
    for i, c in enumerate(criteria_labels):
        with cols[i%3]:
            choice = st.selectbox(f"{c}", list(criteria_dropdown[c].keys()), key=f"new_{c}")
            new_vals.append(choice)
    if st.button("Tambah Alternatif"):
        if new_name.strip() == "":
            st.error("Nama alternatif tidak boleh kosong.")
        elif new_name in st.session_state.alternatives:
            st.error("Nama sudah ada.")
        else:
            st.session_state.alternatives[new_name] = new_vals
            st.success(f"Alternatif '{new_name}' ditambahkan.")

    st.markdown("---")
    st.subheader("✏️ Edit / 🗑️ Hapus Alternatif")
    sel = st.selectbox("Pilih Alternatif", options=list(st.session_state.alternatives.keys()))
    if sel:
        current = st.session_state.alternatives[sel].copy()
        cols2 = st.columns(3)
        new_edit = []
        for i, c in enumerate(criteria_labels):
            with cols2[i%3]:
                val = st.selectbox(f"{c}", list(criteria_dropdown[c].keys()), index=list(criteria_dropdown[c].keys()).index(current[i]), key=f"edit_{c}")
                new_edit.append(val)
        coldel, colsave = st.columns([1,1])
        with coldel:
            if st.button("Hapus Alternatif"):
                del st.session_state.alternatives[sel]
                st.success("Alternatif dihapus.")
        with colsave:
            if st.button("Simpan Perubahan"):
                st.session_state.alternatives[sel] = new_edit
                st.success("Perubahan disimpan.")

# ---------- MENU: HASIL PERHITUNGAN ----------
elif menu == "Hasil Perhitungan":
    st.markdown("<div class='card'><h3>📈 Hasil Perhitungan</h3><div class='small-muted'>Pilih metode dan lihat ranking + grafik</div></div>", unsafe_allow_html=True)
    if len(st.session_state.alternatives) == 0:
        st.warning("Tidak ada data alternatif. Tambahkan di menu Data & CRUD.")
    else:
        names = list(st.session_state.alternatives.keys())
        values = []
        for name in names:
            row = st.session_state.alternatives[name]
            numeric = [criteria_dropdown[criteria_labels[i]][row[i]] for i in range(len(criteria_labels))]
            values.append(numeric)

        # calculate
        saw_scores, saw_norm = calculate_saw(values, st.session_state.weights)
        wp_S, wp_V = calculate_wp(values, st.session_state.weights)

        option = st.radio("Tampilkan", ["Perbandingan SAW & WP", "SAW saja", "WP saja"], horizontal=True)
        if option == "SAW saja":
            df = pd.DataFrame({
                "Alternatif": names,
                "Skor SAW": saw_scores,
            })
            df["Ranking SAW"] = df["Skor SAW"].rank(ascending=False, method="min").astype(int)
            df = df.sort_values(by="Skor SAW", ascending=False).reset_index(drop=True)
        elif option == "WP saja":
            df = pd.DataFrame({
                "Alternatif": names,
                "Skor WP (V)": wp_V,
            })
            df["Ranking WP"] = df["Skor WP (V)"].rank(ascending=False, method="min").astype(int)
            df = df.sort_values(by="Skor WP (V)", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame({
                "Alternatif": names,
                "Skor SAW": saw_scores,
                "Ranking SAW": pd.Series(saw_scores).rank(ascending=False, method="min").astype(int),
                "Skor WP (V)": wp_V,
                "Ranking WP": pd.Series(wp_V).rank(ascending=False, method="min").astype(int),
            })
            df = df.sort_values(by=["Skor SAW"], ascending=False).reset_index(drop=True)

        # display nicely (HTML table)
        def render_table(df):
            best_saw = None
            best_wp = None
            if "Skor SAW" in df.columns:
                best_saw = df["Skor SAW"].max()
            if "Skor WP (V)" in df.columns:
                best_wp = df["Skor WP (V)"].max()

            html = "<table class='table-dark'>"
            # header
            html += "<tr>"
            for col in df.columns:
                html += f"<th>{col}</th>"
            html += "</tr>"
            # rows
            for _, row in df.iterrows():
                is_best = False
                # highlight if best in either
                if best_saw is not None and "Skor SAW" in df.columns and row.get("Skor SAW") == best_saw:
                    row_class = "best"
                elif best_wp is not None and "Skor WP (V)" in df.columns and row.get("Skor WP (V)") == best_wp:
                    row_class = "best"
                else:
                    row_class = ""
                html += f"<tr class='{row_class}'>"
                for col in df.columns:
                    v = row[col]
                    if isinstance(v, float):
                        html += f"<td>{v:.4f}</td>"
                    else:
                        html += f"<td>{v}</td>"
                html += "</tr>"
            html += "</table>"
            return html

        st.markdown(render_table(df), unsafe_allow_html=True)

        # chart area
        st.markdown("---")
        st.subheader("Visualisasi Perbandingan")
        fig, ax = plt.subplots(figsize=(8,4))
        x = np.arange(len(names))
        ax.bar(x - 0.15, saw_scores, width=0.3, label="SAW", color="#06b6d4")
        ax.bar(x + 0.15, wp_V, width=0.3, label="WP (V)", color="#06b6d4", alpha=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, color="#dff7fb")
        ax.set_ylabel("Skor", color="#dff7fb")
        ax.set_facecolor("#07101a")
        fig.patch.set_facecolor('#0f1720')
        ax.spines['bottom'].set_color('#24313a')
        ax.spines['top'].set_color('#0f1720')
        ax.spines['left'].set_color('#24313a')
        ax.spines['right'].set_color('#0f1720')
        ax.legend(facecolor="#07101a", edgecolor="#07101a", labelcolor="#dff7fb")
        st.pyplot(fig)

        st.markdown("<div class='small-muted'>Tip: Klik menu 'Pengaturan Bobot' untuk mengubah bobot dan lihat dampaknya di sini.</div>", unsafe_allow_html=True)