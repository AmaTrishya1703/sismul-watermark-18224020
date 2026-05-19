"""
Watermarking + JPEG Compression Robustness Evaluator
=====================================================
Tugas:
1. Sisipkan watermark biner ke foto menggunakan LSB embedding
2. Kompres foto ke JPEG dengan berbagai Quality Factor (QF)
3. Ekstrak watermark dari setiap hasil kompresi
4. Evaluasi ketahanan watermark menggunakan NCC

Cara pakai:
    python watermarking_jpeg.py --host foto_wajah.jpg --watermark watermark.png
    python watermarking_jpeg.py  # pakai gambar dummy jika tidak ada file
"""

import io
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ===========================================================================
# 1. UTILITY — Buat gambar dummy jika tidak ada file asli
# ===========================================================================

def create_dummy_host(size=(512, 512)) -> np.ndarray:
    """Buat foto wajah dummy (gradasi warna) sebagai host image."""
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    # Gradient sederhana sebagai placeholder "foto"
    for y in range(size[1]):
        r = int(220 * (1 - y / size[1]))
        g = int(180 * (y / size[1]))
        b = 150
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    # Tambahkan teks supaya ada tekstur
    draw.ellipse([150, 100, 362, 350], fill=(210, 170, 130))
    draw.ellipse([200, 130, 312, 220], fill=(255, 220, 180))
    draw.text((180, 380), "Host Image", fill=(50, 50, 50))
    return np.array(img)


def create_dummy_watermark(size=(128, 128)) -> np.ndarray:
    """Buat watermark biner berbentuk pola sederhana."""
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    # Buat teks "WM" dan kotak sebagai watermark
    draw.rectangle([10, 10, 118, 118], outline=255, width=4)
    draw.text((30, 45), "WM", fill=255)
    draw.ellipse([40, 20, 88, 60], fill=255)
    return np.array(img)


# ===========================================================================
# 2. PREPROCESSING — Load & resize watermark
# ===========================================================================

def load_and_prepare(host_path: str | None, wm_path: str | None):
    """
    Load host image dan watermark.
    Jika path None, gunakan gambar dummy.
    Kembalikan (host_array RGB, wm_binary 0/1, wm_shape).
    """
    # --- Host image ---
    if host_path:
        host_img = Image.open(host_path).convert("RGB")
        host_array = np.array(host_img)
        print(f"[✓] Host image loaded: {host_path} | ukuran {host_img.size}")
    else:
        host_array = create_dummy_host()
        print("[i] Menggunakan host image DUMMY (512x512)")

    H, W = host_array.shape[:2]

    # --- Watermark ---
    if wm_path:
        wm_img = Image.open(wm_path).convert("L")
        print(f"[✓] Watermark loaded: {wm_path}")
    else:
        wm_img = Image.fromarray(create_dummy_watermark())
        print("[i] Menggunakan watermark DUMMY (128x128)")

    # Resize watermark ke 1/4 ukuran host (agar muat di pojok)
    wm_w = W // 4
    wm_h = H // 4
    wm_resized = wm_img.resize((wm_w, wm_h), Image.LANCZOS)
    wm_gray = np.array(wm_resized)

    # Binarisasi: nilai > 127 → 1, lainnya → 0
    wm_binary = (wm_gray > 127).astype(np.uint8)

    print(f"[✓] Watermark di-resize ke {wm_w}×{wm_h}, binarisasi selesai")
    return host_array, wm_binary


# ===========================================================================
# 3. EMBED — Sisipkan watermark ke LSB channel Merah
# ===========================================================================

def embed_watermark(host_array: np.ndarray, wm_binary: np.ndarray) -> np.ndarray:
    """
    Sisipkan wm_binary ke LSB channel R pada pojok kiri atas host_array.
    Setiap bit watermark menggantikan LSB satu pixel.
    """
    result = host_array.copy()
    h, w = wm_binary.shape

    # Pastikan watermark tidak melebihi dimensi host
    assert h <= result.shape[0] and w <= result.shape[1], \
        "Watermark lebih besar dari host image!"

    # Ganti LSB channel R dengan bit watermark
    # (pixel & 0xFE) membersihkan LSB, | wm_bit menyetel LSB
    result[:h, :w, 0] = (result[:h, :w, 0] & 0xFE) | wm_binary

    return result


# ===========================================================================
# 4. COMPRESS — Simpan sebagai JPEG dengan quality factor tertentu
# ===========================================================================

def compress_to_jpeg(watermarked_array: np.ndarray, quality: int) -> np.ndarray:
    """
    Kompres gambar ke JPEG in-memory dengan quality factor tertentu,
    lalu decode kembali ke numpy array RGB.
    """
    img = Image.fromarray(watermarked_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    file_size_kb = buf.tell() / 1024
    buf.seek(0)
    compressed = np.array(Image.open(buf).convert("RGB"))
    return compressed, file_size_kb


# ===========================================================================
# 5. EXTRACT — Ekstrak watermark dari LSB channel Merah
# ===========================================================================

def extract_watermark(compressed_array: np.ndarray, wm_shape: tuple) -> np.ndarray:
    """
    Ekstrak watermark dari LSB channel R pada pojok kiri atas.
    Mengembalikan array biner (0/1) seukuran wm_shape.
    """
    h, w = wm_shape
    extracted = compressed_array[:h, :w, 0] & 1  # ambil LSB
    return extracted.astype(np.uint8)


# ===========================================================================
# 6. EVALUATE — Hitung NCC antara watermark asli vs terekstrak
# ===========================================================================

def compute_ncc(wm_original: np.ndarray, wm_extracted: np.ndarray) -> float:
    """
    Normalized Cross-Correlation (NCC).
    Nilai 1.0  = identik
    Nilai ~0.0 = tidak berkorelasi (watermark rusak)
    Nilai -1.0 = terbalik sempurna
    """
    a = wm_original.astype(np.float64).flatten()
    b = wm_extracted.astype(np.float64).flatten()

    a -= a.mean()
    b -= b.mean()

    numerator = np.dot(a, b)
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


# ===========================================================================
# 7. VISUALIZE — Plot hasil NCC dan tampilkan watermark tiap QF
# ===========================================================================

def visualize_results(wm_binary, results, output_path="hasil_watermarking.png"):
    """
    Buat figure berisi:
    - Grafik NCC vs QF
    - Preview watermark terekstrak untuk beberapa QF
    """
    qf_list   = [r["qf"]  for r in results]
    ncc_list  = [r["ncc"] for r in results]
    kb_list   = [r["kb"]  for r in results]
    wm_list   = [r["wm_extracted"] for r in results]

    # Pilih 5 QF untuk preview watermark
    preview_idx = np.linspace(0, len(results) - 1, 5, dtype=int)

    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle("Evaluasi Ketahanan Watermark terhadap Kompresi JPEG",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 6, figure=fig,
                           hspace=0.45, wspace=0.4,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)

    # --- Baris atas: Grafik NCC ---
    ax_ncc = fig.add_subplot(gs[0, :4])
    ax_ncc.set_facecolor("#1a1a2e")

    colors = ["#00ff88" if n >= 0.7 else "#ff4466" for n in ncc_list]
    bars = ax_ncc.bar(range(len(qf_list)), ncc_list, color=colors, width=0.6, zorder=3)
    ax_ncc.plot(range(len(qf_list)), ncc_list, "o-",
                color="#ffffff", linewidth=1.5, markersize=5, zorder=4)
    ax_ncc.axhline(0.7, color="#ffdd00", linestyle="--", linewidth=1.5,
                   label="Threshold NCC = 0.70", zorder=5)

    ax_ncc.set_xticks(range(len(qf_list)))
    ax_ncc.set_xticklabels([str(q) for q in qf_list], color="white", fontsize=10)
    ax_ncc.set_xlabel("Quality Factor (QF)", color="#aaaaaa", fontsize=11)
    ax_ncc.set_ylabel("NCC", color="#aaaaaa", fontsize=11)
    ax_ncc.set_ylim(-0.05, 1.1)
    ax_ncc.set_title("NCC vs Quality Factor", color="white", fontsize=12, pad=8)
    ax_ncc.tick_params(colors="white")
    ax_ncc.spines[["top", "right"]].set_visible(False)
    ax_ncc.spines[["left", "bottom"]].set_color("#444455")
    ax_ncc.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    ax_ncc.grid(axis="y", color="#333344", linestyle="--", alpha=0.5, zorder=0)

    # Anotasi nilai NCC di atas bar
    for i, (bar, ncc) in enumerate(zip(bars, ncc_list)):
        ax_ncc.text(i, ncc + 0.02, f"{ncc:.2f}",
                    ha="center", va="bottom", color="white", fontsize=8)

    # --- Baris atas kanan: Grafik File Size ---
    ax_kb = fig.add_subplot(gs[0, 4:])
    ax_kb.set_facecolor("#1a1a2e")
    ax_kb.plot(range(len(qf_list)), kb_list, "s-",
               color="#66ccff", linewidth=2, markersize=6)
    ax_kb.fill_between(range(len(qf_list)), kb_list, alpha=0.2, color="#66ccff")
    ax_kb.set_xticks(range(len(qf_list)))
    ax_kb.set_xticklabels([str(q) for q in qf_list], color="white", fontsize=9)
    ax_kb.set_xlabel("Quality Factor (QF)", color="#aaaaaa", fontsize=10)
    ax_kb.set_ylabel("Ukuran File (KB)", color="#aaaaaa", fontsize=10)
    ax_kb.set_title("Ukuran File vs QF", color="white", fontsize=11, pad=8)
    ax_kb.tick_params(colors="white")
    ax_kb.spines[["top", "right"]].set_visible(False)
    ax_kb.spines[["left", "bottom"]].set_color("#444455")
    ax_kb.grid(color="#333344", linestyle="--", alpha=0.5)

    # --- Baris bawah: Preview watermark terekstrak ---
    # Watermark asli
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(wm_binary, cmap="gray", vmin=0, vmax=1)
    ax_orig.set_title("Watermark\nAsli", color="white", fontsize=9)
    ax_orig.axis("off")

    # 5 preview watermark hasil ekstraksi
    for col_idx, r_idx in enumerate(preview_idx):
        ax = fig.add_subplot(gs[1, col_idx + 1])
        ax.imshow(wm_list[r_idx], cmap="gray", vmin=0, vmax=1)
        ncc_val = ncc_list[r_idx]
        status  = "✓" if ncc_val >= 0.7 else "✗"
        color   = "#00ff88" if ncc_val >= 0.7 else "#ff4466"
        ax.set_title(f"QF={qf_list[r_idx]}\nNCC={ncc_val:.2f} {status}",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[✓] Visualisasi disimpan ke: {output_path}")
    plt.show()


# ===========================================================================
# 8. MAIN — Jalankan semua langkah
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Watermarking + JPEG Robustness Evaluator"
    )
    parser.add_argument("--host",      type=str, default=None,
                        help="Path ke foto host (jpg/png). Default: gambar dummy")
    parser.add_argument("--watermark", type=str, default=None,
                        help="Path ke watermark biner (png). Default: pola dummy")
    parser.add_argument("--output",    type=str, default="hasil_watermarking.png",
                        help="Nama file output grafik. Default: hasil_watermarking.png")
    args = parser.parse_args()

    print("=" * 60)
    print("  WATERMARKING + JPEG ROBUSTNESS EVALUATOR")
    print("=" * 60)

    # ---- Step 1: Load gambar ----
    print("\n[STEP 1] Memuat gambar...")
    host_array, wm_binary = load_and_prepare(args.host, args.watermark)

    # ---- Step 2: Embed watermark ----
    print("\n[STEP 2] Menyisipkan watermark (LSB embedding)...")
    watermarked = embed_watermark(host_array, wm_binary)
    Image.fromarray(watermarked).save("watermarked_lossless.png")
    print("[✓] Tersimpan: watermarked_lossless.png")

    # ---- Step 3 & 4: Loop QF dari besar ke kecil ----
    quality_factors = [99, 98, 97, 96, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    print(f"\n[STEP 3 & 4] Kompresi JPEG & Evaluasi NCC")
    print(f"{'QF':>5} | {'Ukuran (KB)':>11} | {'NCC':>8} | Status")
    print("-" * 40)

    results = []
    for qf in quality_factors:
        # Kompres
        compressed, file_kb = compress_to_jpeg(watermarked, qf)

        # Ekstrak watermark
        wm_extracted = extract_watermark(compressed, wm_binary.shape)

        # Hitung NCC
        ncc = compute_ncc(wm_binary, wm_extracted)

        status = "✅ BAGUS" if ncc >= 0.7 else "⚠️  RUSAK"
        print(f"{qf:>5} | {file_kb:>9.1f} KB | {ncc:>8.4f} | {status}")

        results.append({
            "qf":           qf,
            "kb":           file_kb,
            "ncc":          ncc,
            "wm_extracted": wm_extracted,
        })

    # ---- Step 5: Temukan QF kritis ----
    print("\n[STEP 5] Analisis QF kritis...")
    critical_qf = None
    for r in reversed(results):         # dari QF kecil ke besar
        if r["ncc"] >= 0.7:
            critical_qf = r["qf"]
            break

    if critical_qf is not None:
        print(f"[✓] QF minimum yang masih 'BAGUS': QF = {critical_qf}")
    else:
        print("[!] Watermark rusak di semua QF yang diuji.")

    # ---- Step 6: Visualisasi ----
    print("\n[STEP 6] Membuat visualisasi...")
    visualize_results(wm_binary, results, output_path=args.output)

    print("\n[SELESAI] Program selesai.")


if __name__ == "__main__":
    main()