# Benchmark Quick Start Guide

En hızlı şekilde benchmark'ları çalıştırmak için bu rehberi takip edin.

## ⚡ Hızlı Başlangıç (3 Adım)

### 1. Setup
```bash
cd benchmarks
make setup
```

Bu komut:
- ✅ Virtual environment oluşturur
- ✅ Tüm bağımlılıkları yükler
- ✅ Upsonic'i editable mode'da yükler

### 2. API Key
Ana dizinde `.env` dosyası oluşturun:
```bash
cd ..
echo "OPENAI_API_KEY=sk-your-key-here" > .env
cd benchmarks
```

### 3. Çalıştır
```bash
make run
```

Hepsi bu kadar! 🎉

---

## 📚 Diğer Komutlar

### Test Case'leri Göster
```bash
make list
```

### Tüm Testleri Çalıştır
```bash
make run-all  # Dikkat: 5+ dakika sürebilir
```

### Belirli Bir Test
```bash
make run-math           # Math problem
make run-structured     # Structured output
make run-analysis       # Text analysis
```

### Özel İterasyon Sayısı
```bash
make run-iterations N=10  # 10 iterasyon
```

### Sonuçları Göster
```bash
make results
```

### Environment Kontrolü
```bash
make test-env
```

Çıktı:
```
✓ Virtual environment exists
✓ .env file exists
✓ Upsonic installed
```

---

## 🔧 Sorun Giderme

### "Virtual environment not found"
```bash
make setup
```

### ".env file not found"
```bash
cd ..
nano .env  # OPENAI_API_KEY ekle
cd benchmarks
```

### Bağımlılık Hatası
```bash
make install
```

### Her Şeyi Sıfırla
```bash
make clean-all
make setup
```

---

## 📊 Örnek Workflow

```bash
# İlk kurulum
cd benchmarks
make setup
cd .. && echo "OPENAI_API_KEY=sk-xxx" > .env && cd benchmarks

# Hızlı test
make list       # Test case'leri gör
make run        # Basit test çalıştır

# Detaylı analiz
make run-all    # Tüm testleri çalıştır

# Sonuçları görüntüle
make results    # JSON dosyaları listele
cat overhead_analysis/results/*.json | jq .  # JSON içeriği gör

# Temizlik
make clean      # Cache temizle
```

---

## 🎯 Sonuçları Anlama

Benchmark sonuçları şunları gösterir:

**Detailed Comparison Table:**
- Speed Metrics: Mean, Median, Stdev, Min, Max (ms)
- Memory: Object size (bytes)
- Cost: Per iteration ve total cost
- Token Usage: Mean ve total token sayıları

**Three-Way Comparison:**
- Direct: Minimum overhead
- Agent (no prompt): System prompt olmadan
- Agent (with prompt): Default system prompt ile

**Sample Outputs:**
- Her approach'un gerçek cevapları
- Kalite farklarını görebilirsiniz

---

## 💡 İpuçları

1. **İlk çalıştırma daha yavaş**: Model yükleme, cache oluşturma
2. **API maliyeti**: Her test ~$0.00001-0.0001 arası
3. **İterasyon sayısı**: Daha fazla iterasyon = daha güvenilir sonuçlar
4. **Network bağlantısı gerekli**: LLM API çağrıları için

---

## 🆘 Yardım

Tüm komutları görmek için:
```bash
make help
```

Detaylı dokümantasyon için:
- `README.md` - Ana README
- `SETUP.md` - Detaylı kurulum
- `overhead_analysis/README.md` - Proje specific

