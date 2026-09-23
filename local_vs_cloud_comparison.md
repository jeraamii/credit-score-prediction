# Perbandingan Local vs Cloud ML Pipeline

## Local Pipeline (MLflow + Streamlit local)

| Aspek | Keunggulan | Kelemahan |
|---|---|---|
| **Setup** | Cepat dimulai, tidak perlu akun cloud | Terbatas pada resource satu mesin |
| **Biaya** | Gratis (tidak ada tagihan cloud) | Upgrade hardware membutuhkan biaya |
| **Skalabilitas** | Cocok untuk dataset kecil–menengah | Tidak bisa scale horizontal |
| **Privasi data** | Data tidak keluar dari mesin lokal | Tidak ada akses remote tanpa VPN |
| **Debugging** | Mudah inspeksi log, file, dan model | Sulit berbagi hasil dengan tim |
| **MLflow UI** | Berjalan instan di localhost:5000 | Tidak bisa diakses dari luar tanpa tunnel |
| **Deployment** | Streamlit jalan di localhost:8501 | Hanya bisa diakses lokal |
| **Reprodusibilitas** | Environment terkontrol penuh | Bisa terjadi drift antar mesin |

## Cloud Pipeline (AWS SageMaker + S3 + Streamlit Cloud)

| Aspek | Keunggulan | Kelemahan |
|---|---|---|
| **Setup** | Infrastruktur terkelola, tidak perlu konfigurasi hardware | Perlu akun AWS, IAM role, credentials |
| **Biaya** | Pay-per-use, bisa scale down ke nol | Endpoint aktif terus dikenakan biaya |
| **Skalabilitas** | Auto-scaling, distributed training tersedia | Overkill untuk proyek kecil |
| **Privasi data** | Enkripsi enterprise-grade, IAM policy | Data keluar dari premise; perlu compliance review |
| **Debugging** | CloudWatch logs, SageMaker Debugger | Lebih sulit reproduksi error secara lokal |
| **MLflow** | Bisa host di EC2 atau SageMaker Studio | Setup tambahan diperlukan |
| **Deployment** | Public HTTPS endpoint, Streamlit Cloud URL bisa diakses publik | Cold-start latency jika di-scale ke nol |
| **Reprodusibilitas** | Docker container, S3-versioned artifacts | Manajemen container lebih kompleks |

## Kesimpulan

| Kondisi | Rekomendasi |
|---|---|
| Prototyping / belajar / data sensitif | **Local** |
| Produksi / tim / data besar / akses publik | **Cloud (AWS)** |

Praktik terbaik: **kembangkan secara lokal, deploy ke cloud** — persis seperti alur yang diikuti dalam proyek ini (local pipeline → AWS SageMaker → Streamlit Cloud).
