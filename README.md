# Domain Spesifik LLM: QLoRA Finetuning ve RAG Entegrasyonu

Bu projede, galaksidışı astronomide derin öğrenme kullanan çalışmalara gerçekten hakim olabilen tamamen yerel çalışan bir araştırma asistanı geliştirmeyi amaçladım. Sistemi Apple Silicon (Mac Mini M4, 16 GB RAM) üzerinde API maliyeti olmadan çalışacak hibrit bir mimariyle kurdum.

## Sistem Mimarisi

Sistem, genel amaçlı modellerin teknik detaylardaki yetersizliğini ve halüsinasyon riskini aşmak için iki ana bileşeni birleştirir:

* **Fine-Tuning (İnce Ayar):** Qwen2.5-7B modeli, ChatML formatında hazırlanmış 223 soru-cevap çifti ile eğitilerek akademik terminoloji kazandırılmıştır.
* **RAG (Retrieval-Augmented Generation):** 15 adet arXiv makalesini vektör veritabanına indekslenerek, modelin yanıt üretirken doğrudan kaynak metne erişmesi sağlanmıştır.

## Pipeline Bileşenleri

1. **arxiv_tex_data.py:** arXiv üzerinden indirilen LaTeX kaynak kodlarını temizleyerek yapılandırılmış düz metne dönüştürür.
2. **merge_qa.py:** Hazırlanan soru-cevap setlerini birleştirerek eğitim ve test splitlerine ayırır.
3. **finetune.py:** Apple MLX kütüphanesi kullanılarak 4-bit nicelleştirilmiş QLoRA eğitimini gerçekleştirir.
4. **rag.py:** Metinleri BGE-small-en-v1.5 ile vektörleştirir ve ChromaDB üzerinde saklar.
5. **eval_rag.py:** Bellek yönetimini optimize etmek adına iki fazlı (Qwen üretimi ve Llama-3.1 hakemliği) bir değerlendirme hattı çalıştırır.

## Performans Metrikleri

| Metrik | Skor | Açıklama |
| --- | --- | --- |
| Faithfulness | 4.9 / 5 | Cevapların kaynak metne sadakati. |
| Answer Relevance | 4.6 / 5 | Yanıtların kullanıcı sorularını karşılama oranı. |
| Context Recall@3 | %81.6 | Doğru makale parçasına erişim başarısı. |
| Semantic Similarity | 0.77 | Referans cevaplarla anlamsal yakınlık. |

## Kurulum ve Kullanım

### Gereksinimler

* Python 3.11.4
* Apple Silicon işlemcili bir Mac

### Kurulum

```bash
conda create -n paper_llm python=3.11.4 -y && conda activate paper_llm
pip install -r requirements.txt

```

### Çalıştırma Sırası

1. Veriyi Hazırla: `python arxiv_tex_data.py`
2. Soru-Cevap Setini Birleştir: `python merge_qa.py`
3. Eğitimi Başlat: `python finetune.py train`
4. İndeksle: `python rag.py index`
5. Chat Modu: `python rag.py chat`
6. Değerlendir: `python eval_rag.py`

## Not

Bellek yönetimini optimize etmek için değerlendirme aşamasında modeller (Qwen ve Llama) sıralı olarak yüklenip boşaltılır.

