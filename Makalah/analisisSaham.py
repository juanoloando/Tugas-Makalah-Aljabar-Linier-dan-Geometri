import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def dataLoadingnCleansing(dataDir: Path):
    csvs = sorted([p for p in dataDir.glob("*.csv")])
    frames = []
    for csvPath in csvs:
         ticker = csvPath.stem
         df =  pd.read_csv(csvPath)[["date", "close"]]
         df["date"] = pd.to_datetime(df["date"])
         df = df.rename(columns={"close": ticker}).set_index("date")
         frames.append(df)
    
    harga = pd.concat(frames, axis=1, join="outer")
    harga = harga.loc[:, harga.notna().mean() >= 0.7]
    harga = harga.sort_index().ffill().dropna()
    return harga

def persiapanReturn(harga):
    logReturn = np.log(harga / harga.shift(1)).dropna()
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(logReturn)
    return logReturn, scaled_returns

def visualisasiPlotScree(pca, outdir):
    var_exp = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(var_exp)+1), var_exp, color="#4C72B0", label="Varians per PC")
    plt.plot(range(1, len(var_exp)+1), np.cumsum(var_exp), "o--", color="#55A868", label="Kumulatif")
    plt.title("Scree Plot: Penjelasan Varians Risiko")
    plt.xlabel("Komponen Utama (PC)")
    plt.ylabel("Ratio Varians")
    plt.legend()
    plt.savefig(outdir / "hasilScreePlot.png", dpi=150)
    plt.close()

def visualisasiLoadingsPasar(pca, tickers, outdir):
    loadingPC1 = pd.DataFrame({"Saham": tickers, "Loading": pca.components_[0]})
    loadingPC1 = loadingPC1.sort_values("Loading", ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Saham", y="Loading", data=loadingPC1, color="#4C72B0")
    plt.xticks(rotation=90)
    plt.title("PC1 Loadings: Identifikasi Saham Penggerak Pasar")
    plt.savefig(outdir / "hasilLoadingsPasar.png", dpi=150)
    plt.close()

def visualisasiHeatmapLoadings(pca, tickers, outdir):
    loadings = pd.DataFrame(pca.components_[:3, :], index=["PC1 (Market)", "PC2 (Sektor A)", "PC3 (Sektor B)"], columns=tickers)
    plt.figure(figsize=(12, 6))
    sns.heatmap(loadings, cmap="coolwarm", center=0, annot=False)
    plt.title("Heatmap Loadings PC1-PC3: Peta Risiko Sektoral")
    plt.savefig(outdir / "hasilHeatmapLoadings.png", dpi=150)
    plt.close()

def visualisasiPerformaPortofolio(returns, pca, outdir):
    ep_cum_rets = pd.DataFrame(index=returns.index)
    for i in range(3):
        weights = pca.components_[i]
        ep_returns = returns.dot(weights) / np.sum(np.abs(weights))
        ep_cum_rets[f"EP{i+1}"] = (1 + ep_returns).cumprod()
    plt.figure(figsize=(10, 6))
    for col in ep_cum_rets.columns:
        plt.plot(ep_cum_rets.index, ep_cum_rets[col], label=col)
    plt.title("Performa Eigen-Portfolio PC1-PC3")
    plt.xlabel("Tanggal")
    plt.ylabel("Nilai Kumulatif")
    plt.legend()
    plt.savefig(outdir / "hasilPerformaPortofolio.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    dataDir = Path("src/Dataset-Saham-IDX/Saham/LQ45")
    outdir = Path("outputs/hailAnalisisSaham")
    outdir.mkdir(parents=True, exist_ok=True)

    harga = dataLoadingnCleansing(dataDir)
    logReturn, scaled_returns = persiapanReturn(harga)

    pca = PCA()
    pca_scores = pca.fit_transform(scaled_returns)

    visualisasiPlotScree(pca, outdir)
    visualisasiLoadingsPasar(pca, harga.columns.tolist(), outdir)
    visualisasiHeatmapLoadings(pca, harga.columns.tolist(), outdir)
    visualisasiPerformaPortofolio(logReturn, pca, outdir)