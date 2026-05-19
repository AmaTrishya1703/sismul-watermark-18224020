"""
DCT-Based Watermarking + JPEG Robustness Evaluator
===================================================
Metode: Watermark disisipkan ke koefisien DCT mid-frequency
        pada channel Y (luminance) gambar, sehingga tahan
        terhadap kompresi JPEG.

Cara pakai:
    python watermarking_dct.py --host foto_wajah.jpg --watermark watermark.png
    python watermarking_dct.py   # pakai gambar dummy
"""

import io
import argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import dct, idct


# ===========================================================================
# 1. UTILITY — Gambar dummy
# ===========================================================================

def create_dummy_host(size=(512, 512)) -> np.ndarray:
    """Buat foto wajah dummy sebagai host image (RGB)."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        r = int(220 * (1 - y / size[1]))
        g = int(180 * (y / size[1]))
        b = 150
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    draw.ellipse([130, 80, 382, 370], fill=(210, 170, 130))
    draw.ellipse([190, 120, 322, 230], fill=(255, 220, 180))
    draw.ellipse([200, 140, 240, 180], fill=(60, 40, 30))
    draw.ellipse([282, 140, 322, 180], fill=(60, 40, 30))
    draw.arc([210, 220, 310, 270], start=10, end=170, fill=(120, 60, 60), width=3)
    draw.text((190, 390), "Host Image", fill=(50, 50, 50))
    return np.array(img)


def create_dummy_watermark(size=(64, 64)) -> np.ndarray:
    """Buat watermark biner sederhana."""
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 60, 60], outline=255, width=3)
    draw.ellipse([14, 14, 50, 50], fill=255)
    draw.text((18, 22), "WM", fill=0)
    return np.array(img)


# ===========================================================================
# 2. LOAD & PREPARE
# ===========================================================================

def load_and_prepare(host_path, wm_path):
    """Load host image dan watermark, kembalikan array siap pakai."""
    if host_path:
        host_img = Image.open(host_path).convert("RGB")
        host_array = np.array(host_img)
        print(f"[✓] Host image loaded: {host_path} | ukuran {host_img.size}")
    else:
        host_array = create_dummy_host()
        print("[i] Menggunakan host image DUMMY (512×512)")

    H, W = host_array.shape[:2]

    if wm_path:
        wm_img = Image.open(wm_path).convert("L")
        print(f"[✓] Watermark loaded: {wm_path}")
    else:
        wm_img = Image.fromarray(create_dummy_watermark())
        print("[i] Menggunakan watermark DUMMY (64×64)")

    # Watermark dikecilkan agar cocok dengan jumlah blok DCT yang tersedia
    # Jumlah blok 8x8 pada gambar = (H//8) x (W//8)
    max_wm_h = H // 8
    max_wm_w = W // 8
    wm_resized = wm_img.resize((min(64, max_wm_w), min(64, max_wm_h)), Image.LANCZOS)
    wm_binary = (np.array(wm_resized) > 127).astype(np.float64)  # 0.0 atau 1.0

    print(f"[✓] Watermark di-resize ke {wm_binary.shape[1]}×{wm_binary.shape[0]}, binarisasi selesai")
    return host_array, wm_binary


# ===========================================================================
# 3. DCT HELPER — 2D DCT & IDCT per blok
# ===========================================================================

def dct2(block):
    """2D DCT (tipe-II) pada blok 8×8."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    """2D IDCT pada blok 8×8."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# Posisi koefisien mid-frequency dalam blok 8×8 (zigzag order, posisi 3-6)
# Mid-frequency dipilih karena: terlalu rendah → terlihat, terlalu tinggi → hilang saat kompresi
MID_FREQ_POSITIONS = [(1, 2), (2, 1), (3, 0), (2, 2)]


# ===========================================================================
# 4. EMBED — Sisipkan watermark ke koefisien DCT mid-frequency
# ===========================================================================

def embed_watermark_dct(host_array: np.ndarray, wm_binary: np.ndarray,
                        alpha: float = 30.0) -> np.ndarray:
    """
    Sisipkan watermark ke koefisien DCT mid-frequency channel Y (luminance).

    alpha = kekuatan embedding. Makin besar → lebih tahan kompresi,
            tapi makin terlihat perubahan visualnya.
    """
    # Konversi RGB → YCbCr, ambil channel Y
    host_ycbcr = np.array(Image.fromarray(host_array).convert("YCbCr"), dtype=np.float64)
    Y = host_ycbcr[:, :, 0].copy()

    H, W = Y.shape
    wm_h, wm_w = wm_binary.shape

    wm_idx = 0          # indeks bit watermark yang sedang disisipkan
    wm_flat = wm_binary.flatten()
    total_bits = len(wm_flat)

    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            if wm_idx >= total_bits:
                break
            block = Y[row:row+8, col:col+8]
            dct_block = dct2(block)

            # Ambil satu posisi mid-frequency untuk menyimpan satu bit watermark
            pos = MID_FREQ_POSITIONS[wm_idx % len(MID_FREQ_POSITIONS)]
            bit = wm_flat[wm_idx]  # 0.0 atau 1.0

            # Quantization-based embedding:
            # Jika bit=1 → koefisien dibulatkan ke kelipatan alpha yang ganjil
            # Jika bit=0 → koefisien dibulatkan ke kelipatan alpha yang genap
            coeff = dct_block[pos]
            quantized = round(coeff / alpha)
            if bit == 1:
                if quantized % 2 == 0:
                    quantized += 1
            else:
                if quantized % 2 != 0:
                    quantized += 1
            dct_block[pos] = quantized * alpha

            Y[row:row+8, col:col+8] = idct2(dct_block)
            wm_idx += 1
        if wm_idx >= total_bits:
            break

    # Clip agar tetap dalam range [0, 255]
    Y = np.clip(Y, 0, 255)
    host_ycbcr[:, :, 0] = Y

    # Konversi kembali ke RGB
    watermarked_rgb = np.array(
        Image.fromarray(host_ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
    )
    return watermarked_rgb


# ===========================================================================
# 5. COMPRESS — JPEG compression
# ===========================================================================

def compress_to_jpeg(watermarked_array: np.ndarray, quality: int):
    """Kompres ke JPEG in-memory, kembalikan (array RGB, ukuran KB)."""
    img = Image.fromarray(watermarked_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    file_size_kb = buf.tell() / 1024
    buf.seek(0)
    compressed = np.array(Image.open(buf).convert("RGB"))
    return compressed, file_size_kb


# ===========================================================================
# 6. EXTRACT — Ekstrak watermark dari koefisien DCT
# ===========================================================================

def extract_watermark_dct(compressed_array: np.ndarray, wm_shape: tuple,
                          alpha: float = 30.0) -> np.ndarray:
    """
    Ekstrak watermark dari koefisien DCT mid-frequency channel Y.
    Mengembalikan array biner (0/1) seukuran wm_shape.
    """
    comp_ycbcr = np.array(
        Image.fromarray(compressed_array).convert("YCbCr"), dtype=np.float64
    )
    Y = comp_ycbcr[:, :, 0]
    H, W = Y.shape

    wm_h, wm_w = wm_shape
    total_bits = wm_h * wm_w
    extracted_bits = []

    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            if len(extracted_bits) >= total_bits:
                break
            block = Y[row:row+8, col:col+8]
            dct_block = dct2(block)

            idx = len(extracted_bits)
            pos = MID_FREQ_POSITIONS[idx % len(MID_FREQ_POSITIONS)]
            coeff = dct_block[pos]
            quantized = round(coeff / alpha)

            # Genap → bit 0, Ganjil → bit 1
            bit = int(abs(quantized) % 2)
            extracted_bits.append(bit)
        if len(extracted_bits) >= total_bits:
            break

    # Pad jika kurang (bisa terjadi jika gambar terlalu kecil)
    while len(extracted_bits) < total_bits:
        extracted_bits.append(0)

    extracted = np.array(extracted_bits[:total_bits], dtype=np.uint8).reshape(wm_shape)
    return extracted


# ===========================================================================
# 7. NCC — Normalized Cross-Correlation
# ===========================================================================

def compute_ncc(wm_original: np.ndarray, wm_extracted: np.ndarray) -> float:
    """NCC antara watermark asli dan terekstrak. Range: -1 s/d 1."""
    a = wm_original.astype(np.float64).flatten()
    b = wm_extracted.astype(np.float64).flatten()
    a -= a.mean()
    b -= b.mean()
    num = np.dot(a, b)
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(num / den) if den > 1e-10 else 0.0


# ===========================================================================
# 8. VISUALIZE
# ===========================================================================

def visualize_results(wm_binary, results, output_path="hasil_dct_watermarking.png"):
    qf_list  = [r["qf"]  for r in results]
    ncc_list = [r["ncc"] for r in results]
    kb_list  = [r["kb"]  for r in results]
    wm_list  = [r["wm_extracted"] for r in results]

    preview_idx = np.linspace(0, len(results) - 1, 5, dtype=int)

    fig = plt.figure(figsize=(16, 10), facecolor="#0a0f1e")
    fig.suptitle("Evaluasi Ketahanan Watermark DCT terhadap Kompresi JPEG",
                 fontsize=15, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 6, figure=fig,
                           hspace=0.5, wspace=0.4,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)

    # --- Grafik NCC ---
    ax_ncc = fig.add_subplot(gs[0, :4])
    ax_ncc.set_facecolor("#111827")
    colors = ["#00e5ff" if n >= 0.7 else "#ff4d6d" for n in ncc_list]
    bars = ax_ncc.bar(range(len(qf_list)), ncc_list, color=colors, width=0.6, zorder=3)
    ax_ncc.plot(range(len(qf_list)), ncc_list, "o-",
                color="white", linewidth=1.5, markersize=5, zorder=4)
    ax_ncc.axhline(0.7, color="#ffd60a", linestyle="--", linewidth=1.8,
                   label="Threshold NCC = 0.70", zorder=5)
    ax_ncc.set_xticks(range(len(qf_list)))
    ax_ncc.set_xticklabels([str(q) for q in qf_list], color="white", fontsize=10)
    ax_ncc.set_xlabel("Quality Factor (QF)", color="#aaaaaa", fontsize=11)
    ax_ncc.set_ylabel("NCC", color="#aaaaaa", fontsize=11)
    ax_ncc.set_ylim(-0.1, 1.15)
    ax_ncc.set_title("NCC vs Quality Factor (DCT Watermarking)", color="white", fontsize=12, pad=8)
    ax_ncc.tick_params(colors="white")
    ax_ncc.spines[["top", "right"]].set_visible(False)
    ax_ncc.spines[["left", "bottom"]].set_color("#334155")
    ax_ncc.legend(fontsize=9, facecolor="#111827", labelcolor="white", framealpha=0.8)
    ax_ncc.grid(axis="y", color="#1e293b", linestyle="--", alpha=0.7, zorder=0)
    for i, (bar, ncc) in enumerate(zip(bars, ncc_list)):
        ypos = max(ncc + 0.02, 0.03)
        ax_ncc.text(i, ypos, f"{ncc:.2f}",
                    ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

    # --- Grafik Ukuran File ---
    ax_kb = fig.add_subplot(gs[0, 4:])
    ax_kb.set_facecolor("#111827")
    ax_kb.plot(range(len(qf_list)), kb_list, "s-",
               color="#7dd3fc", linewidth=2, markersize=6)
    ax_kb.fill_between(range(len(qf_list)), kb_list, alpha=0.15, color="#7dd3fc")
    ax_kb.set_xticks(range(len(qf_list)))
    ax_kb.set_xticklabels([str(q) for q in qf_list], color="white", fontsize=9)
    ax_kb.set_xlabel("Quality Factor (QF)", color="#aaaaaa", fontsize=10)
    ax_kb.set_ylabel("Ukuran File (KB)", color="#aaaaaa", fontsize=10)
    ax_kb.set_title("Ukuran File vs QF", color="white", fontsize=11, pad=8)
    ax_kb.tick_params(colors="white")
    ax_kb.spines[["top", "right"]].set_visible(False)
    ax_kb.spines[["left", "bottom"]].set_color("#334155")
    ax_kb.grid(color="#1e293b", linestyle="--", alpha=0.7)

    # --- Preview Watermark ---
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(wm_binary, cmap="gray", vmin=0, vmax=1)
    ax_orig.set_title("Watermark\nAsli", color="white", fontsize=9)
    ax_orig.axis("off")

    for col_idx, r_idx in enumerate(preview_idx):
        ax = fig.add_subplot(gs[1, col_idx + 1])
        ax.imshow(wm_list[r_idx], cmap="gray", vmin=0, vmax=1)
        ncc_val = ncc_list[r_idx]
        status  = "✓" if ncc_val >= 0.7 else "✗"
        color   = "#00e5ff" if ncc_val >= 0.7 else "#ff4d6d"
        ax.set_title(f"QF={qf_list[r_idx]}\nNCC={ncc_val:.2f} {status}",
                     color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[✓] Visualisasi disimpan ke: {output_path}")
    plt.show()


# ===========================================================================
# 9. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="DCT Watermarking + JPEG Robustness")
    parser.add_argument("--host",      type=str, default=None,
                        help="Path foto host (jpg/png). Default: dummy")
    parser.add_argument("--watermark", type=str, default=None,
                        help="Path watermark biner (png). Default: dummy")
    parser.add_argument("--alpha",     type=float, default=30.0,
                        help="Kekuatan embedding DCT. Default: 30.0")
    parser.add_argument("--output",    type=str, default="hasil_dct_watermarking.png",
                        help="Nama file output grafik")
    args = parser.parse_args()

    print("=" * 60)
    print("  DCT WATERMARKING + JPEG ROBUSTNESS EVALUATOR")
    print("=" * 60)

    # Step 1: Load
    print("\n[STEP 1] Memuat gambar...")
    host_array, wm_binary = load_and_prepare(args.host, args.watermark)

    # Step 2: Embed
    print(f"\n[STEP 2] Menyisipkan watermark (DCT embedding, alpha={args.alpha})...")
    watermarked = embed_watermark_dct(host_array, wm_binary, alpha=args.alpha)
    Image.fromarray(watermarked).save("watermarked_dct_lossless.png")
    print("[✓] Tersimpan: watermarked_dct_lossless.png")

    # Step 3 & 4: Loop QF
    quality_factors = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    print(f"\n[STEP 3 & 4] Kompresi JPEG & Evaluasi NCC")
    print(f"{'QF':>5} | {'Ukuran (KB)':>11} | {'NCC':>8} | Status")
    print("-" * 42)

    results = []
    for qf in quality_factors:
        compressed, file_kb = compress_to_jpeg(watermarked, qf)
        wm_extracted = extract_watermark_dct(compressed, wm_binary.shape, alpha=args.alpha)
        ncc = compute_ncc(wm_binary, wm_extracted)
        status = "✅ BAGUS" if ncc >= 0.7 else "⚠️  RUSAK"
        print(f"{qf:>5} | {file_kb:>9.1f} KB | {ncc:>8.4f} | {status}")
        results.append({"qf": qf, "kb": file_kb, "ncc": ncc, "wm_extracted": wm_extracted})

    # Step 5: QF kritis
    print("\n[STEP 5] Analisis QF kritis...")
    critical_qf = None
    for r in reversed(results):
        if r["ncc"] >= 0.7:
            critical_qf = r["qf"]
            break

    if critical_qf:
        print(f"[✓] QF minimum yang masih BAGUS (NCC ≥ 0.7): QF = {critical_qf}")
    else:
        print("[!] Watermark rusak di semua QF. Coba naikkan nilai --alpha (misal: --alpha 50)")

    # Step 6: Visualisasi
    print("\n[STEP 6] Membuat visualisasi...")
    visualize_results(wm_binary, results, output_path=args.output)

    print("\n[SELESAI] Program selesai.")


if __name__ == "__main__":
    main()