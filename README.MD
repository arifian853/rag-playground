## 📘 RAG Parameter Exploration Assignment

Pada tugas ini kamu akan bereksperimen dengan **5 parameter utama Retrieval-Augmented Generation (RAG)** + **opsi model Groq**.

Kamu akan:

✅ memahami efek tuning RAG

✅ mengamati perubahan kualitas vs kecepatan

✅ belajar trade-off retrieval & generation

✅ menulis insight praktis dari eksperimen

> Tools sudah disediakan. Kamu **tidak perlu ngoding**, hanya ubah nilai parameter di file Python & restart server.

**Notes:** 

**- Ikuti video di LMS jika kamu kebingungan.**

**- Sediakan penyimpanan at least 2GB, bakalan ada download model yang cukup besar ke laptopmu.**

**- Hapus proyek ini kalau sudah selesai, biar laptop mu tidak tercekik**

**- Ketika proyek ini sudah diisi, sudah dikerjakan dibawah, maka upload ke GitHub.**

---

## ⚙️ Parameter yang diuji

### 1️⃣ Ukuran chunk

```python
CHUNK_SIZE = 300
```

**Fungsi**
Mengatur seberapa besar potongan teks untuk embedding.

| Arah tuning               | Efek                                              |
| ------------------------- | ------------------------------------------------- |
| **Lebih besar** (400–600) | ? (jawab) |
| **Lebih kecil** (150–250) | ? (jawab) |

**Pertanyaan**

* Chunk besar atau kecil lebih tepat untuk dokumen yang kamu tes?
* Jawab : 

---

### 2️⃣ Kandidat awal vector search

```python
TOP_K_INITIAL = 20
```

**Fungsi**
Berapa dokumen paling mirip yang diambil sebelum filtering.

| Arah tuning         | Efek                                                |
| ------------------- | --------------------------------------------------- |
| **Naikkan** (30–50) | ? (jawab) |
| **Turunkan** (5–10) | ? (jawab) |

**Pertanyaan**

* Apakah top-k kecil cukup untuk dataset kamu?
* Jawab : 

---

### 3️⃣ Konten final ke LLM

```python
FINAL_TOP_M = 5
```

**Fungsi**
Jumlah konteks yang akhirnya dimasukkan ke prompt LLM.

| Arah tuning        | Efek                                        |
| ------------------ | ------------------------------------------- |
| **Naikkan** (6–8)  | ? (jawab) |
| **Turunkan** (2–3) | ? (jawab) |

**Pertanyaan**

* Benarkah “semakin banyak konteks semakin bagus”?
* Jawab : 

---

### 4️⃣ Cross-Encoder Re-ranker

```python
USE_RERANKER = True
```

**Fungsi**
Menilai ulang (rerank) dokumen hasil embedding.

| Mode    | Efek                                    |
| ------- | --------------------------------------- |
| `True`  | ? (jawab) |
| `False` | ? (jawab) |

**Pertanyaan**

* Pada query jenis apa reranker paling terasa manfaatnya?
* Jawab : 

---

### 5️⃣ Temperatur generasi

```python
TEMPERATURE = 0.2
```

**Fungsi**
Mengontrol variasi / kreativitas jawaban.

| Value   | Efek                                  |
| ------- | ------------------------------------- |
| 0.0–0.2 | ? (jawab) |
| 0.5–0.7 | ? (jawab) |

**Pertanyaan**

* Untuk QA berbasis dokumen perusahaan, temperatur ideal berapa?
* Jawab : 

---

### 6️⃣ Model LLM Groq

```python
GROQ_MODEL = "llama-3.1-8b-instant"
# alternatif:
# gpt-oss-120b
# llama-3.3-70b-versatile
```

| Pilihan                  | Efek                                |
| ------------------------ | ----------------------------------- |
| Model kecil (8B)         | ? (jawab) |
| Model besar (70B / 120B) | ? (jawab) |

**Pertanyaan**

* Apakah peningkatan kualitas sebanding dengan penambahan waktu?
* Jawab : 

---

## Tips

* Pakai PDF minimal 1 halaman, maksimal 10 halaman, biar gak berat.
* Coba pertanyaan definisi, konteks, dan reasoning
* Catat kapan sistem salah & mengapa
* Prioritaskan kualitas hasil dalam laporan

---
