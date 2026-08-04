import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set Tampilan Visualiasi
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

# 1. PERSIAPAN FOLDER OUTPUT
os.makedirs("output", exist_ok=True)

# 2. LOAD DATASET EXCEL
data_path = os.path.join("data", "HR Data.xlsx")
print("=" * 60)
print(f"📂 Membaca dataset dari: {data_path}")
print("=" * 60)

try:
    df = pd.read_excel(data_path)
except Exception as e:
    print(
        f"❌ Gagal membaca file! Pastikan file 'HR Data.xlsx' ada di folder 'data/'. Error: {e}"
    )
    exit()

# ---------------------------------------------------------
# 3. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
print("\n🧹 MEMULAI PROSES DATA CLEANING & PREPROCESSING...\n")

# a. Cek Dimensi Awal
print(f"• Ukuran Data Awal       : {df.shape[0]} baris, {df.shape[1]} kolom")

# b. Hapus Baris Duplikat
duplicate_rows = df.duplicated().sum()
if duplicate_rows > 0:
    df = df.drop_duplicates()
    print(f"• Menghapus Duplikat     : {duplicate_rows} baris duplikat dihapus")
else:
    print("• Menghapus Duplikat     : Tidak ada baris duplikat")

# c. Pembersihan String/Spasi Kosong pada Kolom Teks
string_cols = df.select_dtypes(include=["object"]).columns
for col in string_cols:
    df[col] = df[col].astype(str).str.strip()

# d. Penanganan Missing Values (Nilai Kosong)
missing_series = df.isnull().sum()
missing_cols = missing_series[missing_series > 0]

if len(missing_cols) > 0:
    print("\n⚠️ Ditemukan Missing Values pada kolom:")
    for col, count in missing_cols.items():
        print(f"  - {col}: {count} nilai kosong ({count/len(df)*100:.2f}%)")

    # Strategi Pengisian Missing Values:
    # - Kolom Angka/Numerik diisi dengan Median
    # - Kolom Teks/Kategorikal diisi dengan Modus (Nilai terbanyak)
    num_missing = (
        df[missing_cols.index].select_dtypes(include=[np.number]).columns
    )
    cat_missing = (
        df[missing_cols.index].select_dtypes(include=["object"]).columns
    )

    for col in num_missing:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    for col in cat_missing:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    print("  ✅ Missing values berhasil diatasi!")
else:
    print("• Missing Values         : Tidak ditemukan nilai kosong (Data Lengkap)")

# e. Hapus Kolom Redundan / Nilai Konstan / ID
drop_cols = ["Standard Hours", "Over18", "Employee Count", "emp no"]
existing_drop_cols = [c for c in drop_cols if c in df.columns]
if existing_drop_cols:
    df.drop(columns=existing_drop_cols, inplace=True)
    print(f"• Menghapus Kolom Redundan: {existing_drop_cols}")

# f. Standardisasi Target Variable (Attrition)
# Membuat kolom kuantitatif (1 = Attrition/Keluar, 0 = Stay/Bertahan)
if "CF_attrition label" in df.columns:
    df["Attrition_Numeric"] = df["CF_attrition label"].apply(
        lambda x: 1 if str(x).lower() in ["yes", "1", "attrition", "exited"] else 0
    )
elif "Attrition" in df.columns:
    df["Attrition_Numeric"] = df["Attrition"].apply(
        lambda x: 1 if str(x).lower() in ["yes", "1"] else 0
    )
else:
    raise KeyError("Kolom Attrition tidak ditemukan pada dataset!")

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING (Membuat Kolom Baru)
# ---------------------------------------------------------
print("\n⚙️ MELAKUKAN FEATURE ENGINEERING...")

# 1. Ratio Masa Kerja di Perusahaan dibanding Total Pengalaman Kerja
if "Total Working Years" in df.columns and "Years At Company" in df.columns:
    df["Tenure_Ratio"] = np.where(
        df["Total Working Years"] > 0,
        df["Years At Company"] / df["Total Working Years"],
        0,
    )

# 2. Pendapatan per Tahun Pengalaman
if "Monthly Income" in df.columns and "Total Working Years" in df.columns:
    df["Income_Per_Working_Year"] = np.where(
        df["Total Working Years"] > 0,
        df["Monthly Income"] / df["Total Working Years"],
        df["Monthly Income"],
    )

# 3. Kategori Kelompok Usia (Age Group)
if "Age" in df.columns:
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
    )

print("  ✅ Berhasil menambahkan fitur baru: Tenure_Ratio, Income_Per_Working_Year, Age_Group")

# ---------------------------------------------------------
# 5. RINGKASAN EKSEKUTIF / METRIK UTAMA (KPIs)
# ---------------------------------------------------------
total_emp = len(df)
total_attrition = df["Attrition_Numeric"].sum()
attrition_rate = (total_attrition / total_emp) * 100

print("\n" + "=" * 60)
print("📊 RINGKASAN EKSEKUTIF HR ANALYTICS")
print("=" * 60)
print(f"• Total Karyawan              : {total_emp:,}")
print(f"• Karyawan Attrition (Resign) : {total_attrition:,}")
print(f"• Karyawan Bertahan (Stay)    : {total_emp - total_attrition:,}")
print(f"• Overall Attrition Rate      : {attrition_rate:.2f}%\n")

# ---------------------------------------------------------
# 6. VISUALISASI DATA (EDA) & SIMPAN GRAFIK
# ---------------------------------------------------------
print("🎨 MEMBUAT VISUALISASI ANALISIS DATA...")

# --- Plot 1: Attrition berdasarkan Departemen ---
if "Department" in df.columns:
    plt.figure(figsize=(9, 5))
    dept_df = (
        df.groupby("Department")["Attrition_Numeric"]
        .agg(["mean", "count"])
        .reset_index()
    )
    dept_df["rate"] = dept_df["mean"] * 100
    dept_df = dept_df.sort_values(by="rate", ascending=False)

    ax = sns.barplot(
        data=dept_df,
        x="Department",
        y="rate",
        palette="Reds_r",
        hue="Department",
        legend=False,
    )
    plt.title(
        "1. Persentase Attrition Berdasarkan Departemen", fontsize=13, fontweight="bold", pad=15
    )
    plt.ylabel("Tingkat Attrition (%)")
    plt.xlabel("Departemen")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig("output/1_attrition_by_department.png", dpi=300)
    plt.close()

# --- Plot 2: Pengaruh OverTime terhadap Attrition ---
if "Over Time" in df.columns:
    plt.figure(figsize=(7, 5))
    ot_df = (
        df.groupby("Over Time")["Attrition_Numeric"].mean().reset_index()
    )
    ot_df["rate"] = ot_df["Attrition_Numeric"] * 100

    ax = sns.barplot(
        data=ot_df,
        x="Over Time",
        y="rate",
        palette="viridis",
        hue="Over Time",
        legend=False,
    )
    plt.title(
        "2. Dampak Kerja Lembur (OverTime) terhadap Attrition",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Tingkat Attrition (%)")
    plt.xlabel("Status Lembur (Over Time)")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig("output/2_attrition_by_overtime.png", dpi=300)
    plt.close()

# --- Plot 3: Distribusi Gaji Bulanan (Monthly Income) vs Attrition ---
if "Monthly Income" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="CF_attrition label" if "CF_attrition label" in df.columns else "Attrition",
        y="Monthly Income",
        palette="Set2",
    )
    plt.title(
        "3. Distribusi Monthly Income Karyawan (Stay vs Attrition)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Status Attrition")
    plt.ylabel("Gaji Bulanan (Monthly Income)")
    plt.tight_layout()
    plt.savefig("output/3_income_vs_attrition.png", dpi=300)
    plt.close()

# --- Plot 4: Heatmap Korelasi Fitur Utama ---
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
# Pilih fitur numerik penting
cols_for_corr = [
    c
    for c in [
        "Attrition_Numeric",
        "Age",
        "Monthly Income",
        "Total Working Years",
        "Years At Company",
        "Years In Current Role",
        "Years Since Last Promotion",
        "Tenure_Ratio",
        "Environment Satisfaction",
        "Job Satisfaction",
        "Work Life Balance",
    ]
    if c in numeric_cols
]

corr_matrix = df[cols_for_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title(
    "4. Correlation Heatmap Fitur-Fitur Utama Karyawan",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()
plt.savefig("output/4_correlation_heatmap.png", dpi=300)
plt.close()

# --- Plot 5: Attrition Berdasarkan Kelompok Umur (Age Group) ---
if "Age_Group" in df.columns:
    plt.figure(figsize=(8, 5))
    age_df = df.groupby("Age_Group", observed=False)["Attrition_Numeric"].mean().reset_index()
    age_df["rate"] = age_df["Attrition_Numeric"] * 100

    ax = sns.barplot(
        data=age_df,
        x="Age_Group",
        y="rate",
        palette="Blues_d",
        hue="Age_Group",
        legend=False,
    )
    plt.title(
        "5. Tingkat Attrition Berdasarkan Kelompok Usia",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Tingkat Attrition (%)")
    plt.xlabel("Kelompok Usia")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig("output/5_attrition_by_age_group.png", dpi=300)
    plt.close()

print("\n✨ PROSES SELESAI!")
print("📁 5 Grafik visualisasi hasil analisis tersimpan di folder 'output/':")
print("   1. 1_attrition_by_department.png")
print("   2. 2_attrition_by_overtime.png")
print("   3. 3_income_vs_attrition.png")
print("   4. 4_correlation_heatmap.png")
print("   5. 5_attrition_by_age_group.png")
print("=" * 60)
