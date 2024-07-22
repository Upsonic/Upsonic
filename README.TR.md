<p align="center">

<p align="center">
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/deeeb463-c161-4fc6-8407-71c3d8b7defe" alt="Logo"  >
  </a>
  <br>
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/36714716-6990-40b0-84d5-cd7432811bcb" alt="Logo"  >
  </a>

  <h3 align="center">GPT Computer Assistant</h3>
  <p align="center">
    <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>
  </p>

  <p align="center">
    gpt-4o for windows, macos and ubuntu
    <br />
   <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki"><strong>Dökümantasyon</strong></a>
   .
    <a href="https://github.com/onuratakan/gpt-computer-assistant/#Capabilities"><strong>Yeteneklerini Keşfet »</strong></a>
    <br />
    </p>
    <br>
    <p align="center">
     <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki">
   <img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="windows">
   </a>
   <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki">
   <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white" alt="macos">
   </a>
    <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki">
   <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="linux">
   </a>
  <br> 

    
  <p align="center">
  <a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
  </a>
  .
  <img src="https://static.pepy.tech/personalized-badge/gpt-computer-assistant?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20Downloads" alt="pypi_downloads">
  </p>

  <p align="center">
   <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/twitter/follow/GPTCompAsst?style=social" width=160></a>
</p>
<br>


|[ENGLISH](README.md)|[简体中文](README.zh_CN.md)|[正體中文](README.zh_TW.md)|TÜRKÇE|

# GPT Bilgisayar Asistanı

Merhaba, bu ChatGPT MacOS uygulamasının Windows ve Linux için alternatif bir çalışmasıdır. Bu şekilde, taze ve stabil bir çalışma sunuyoruz. Python kütüphanesi olarak kurulumu oldukça kolaydır, ancak ilerleyen zamanlarda .exe formatında doğrudan kurulum betikleri sağlayacak bir iş akışı hazırlayacağız.

Powered by [**Upsonic Tiger 🐅**](https://github.com/Upsonic/Tiger) - LLM ajanları için bir işlev merkezi.

## Kurulum ve Çalıştırma

Python 3.9 veya üstü gereklidir.

```console
pip3 install 'gpt-computer-assistant[base]'
```

```console
computerassistant
```

### Uyandırma Komutu(Wake Word) | YENİ

<details>

Pvporcupine entegrasyonunu ekledik. Bu özelliği kullanmak için ek bir kütüphane kurmanız gerekiyor:

```console
pip3 install 'gpt-computer-assistant[wakeword]'
```

Sonrasında, lütfen [Pvporcupine](https://picovoice.ai/) API anahtarınızı girin ve uyandırma komutu özelliğini etkinleştirin.

</details>

<p align="center">
<br>
  <br>
  <br>

</p>

<p align="center">
<a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/5c6b7063-3d9b-4ea6-befa-ce15d69fcd43" alt="Logo"  >
  </a>
</p>

### Ajan Altyapısı

Bu şekilde `crewai` ajanları oluşturabilir ve onları gpt-computer-assistant arayüzü ve araçları içinde kullanabilirsiniz.

```console
pip3 install 'gpt-computer-assistant[base]'
pip3 install 'gpt-computer-assistant[agentic]'
```

```python
from gpt_computer_assistant import Agent, start

manager = Agent(
  role='Proje Yöneticisi',
  goal='proje ihtiyaçlarını anlar ve kodlayıcıya yardımcı olur',
  backstory="""Büyük bir şirkette bir yöneticisiniz.""",
)

coder = Agent(
  role='Kıdemli Python Geliştirici',
  goal='Python scriptleri yazmak ve panoya kopyalamak',
  backstory="""Büyük bir şirkette bir Python geliştiricisisiniz.""",
)


start()
```



<p align="center">
<a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/c78f3460-6660-4da6-8941-a8ac5cfc1191" alt="Logo"  >
  </a>
</p>

### Özel Araçlar Ekleme

Artık agentic altyapı ve asistan işlemlerinde çalışan özel araçlar ekleyebilirsiniz.


```python
from gpt_computer_assistant import Tool, start

@Tool
def toplam_aracı(ilk_sayı: int, ikinci_sayı: int) -> str:
    """İki sayıyı toplamanız gerektiğinde kullanışlıdır."""
    return ilk_sayı + ikinci_sayı


start()
```






<p align="center">
<a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/65b5fe7c-c0e1-40e9-9447-f41cd4f369a3" alt="Logo"  >
  </a>
</p>


### API | YENİ

Artık GPT Bilgisayar Asistanınızı uzaktan kullanabilirsiniz! GUI hala aktif, bunun için birkaç adım bulunmaktadır:

```console
pip3 install 'gpt-computer-assistant[base]'
pip3 install 'gpt-computer-assistant[api]'
```

```console
computerassistant --api
```


```python
from gpt_computer_assistant.remote import remote

output = remote.input("Merhaba, bugün nasılsın?", screen=False, talk=False)
print(output)

remote.just_screenshot()

remote.talk("TTS test")

# Other Functionalities
remote.reset_memory()
remote.profile("default")

remote.enable_predefined_agents()
remote.disable_predefined_agents()

remote.enable_online_tools()
remote.disable_online_tools()
```






<p align="center">
<br>
  <br>
  <br>
  <br>
  <br>
</p>

<p align="center">
<br>
  <br>
  <br>
</p>


https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/26ae3624-e619-44d6-9b04-f39cf1ac1f8f


## Kullanım
![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/37d34745-ae4b-4b37-9bfa-aec070c97897)



### Kullanım Alanları

<table>
  <tr>
    <td><img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/b4a4f11e-5588-4656-b5d7-b612a9a2855b" alt="Take Meeting Notes" width="500"/></td>
    <td><img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/49eeac70-b33a-4ec4-8125-64127621ed62" alt="Daily Assistant" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/10b69a18-033c-4d81-8ac9-f4e3c65b59c3" alt="Read Docs" width="500"/></td>
    <td><img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/0f483bae-ffaf-4311-8653-c0dc64fb5ebe" alt="Coding Assistant" width="500"/></td>   

  </tr>
</table>






## Yol Haritası 
| Özellik                            | Durum        | Hedef Çeyrek    |
|------------------------------------|--------------|----------------|
| Sohbet Geçmişini Temizle            | Tamamlandı   | 2024 Q2        |
| Uzun Ses Desteği (20mb Bölme)       | Tamamlandı   | 2024 Q2        |
| Metin Girişleri                     | Tamamlandı   | 2024 Q2        |
| Sadece Metin Modu (Konuşmayı Sustur) | Tamamlandı  | 2024 Q2        |
| Profil Ekleme (Farklı Sohbetler)    | Tamamlandı   | 2024 Q2        |
| Asistan Durumu Hakkında Daha Fazla Geri Bildirim | Tamamlandı | 2024 Q2        |
| Yerel Model Görüntü ve Metin (Ollama ve görüntü modelleri ile) | Tamamlandı | 2024 Q2        |
| **Özelleştirilebilir Ajan Altyapısı** | Tamamlandı   | 2024 Q2        |
| Groq Modellerini Destekleme         | Tamamlandı   | 2024 Q2        |
| **Özel Araçlar Ekleme**             | Tamamlandı   | 2024 Q2        |
| Ekrandaki bir şeye tıklama (metin ve simge) | Tamamlandı | 2024 Q2        |
| Yeni Kullanıcı Arayüzü              | Tamamlandı   | 2024 Q2        |
| Yerel Uygulamalar, exe, dmg         | Başarısız (Agentic Altyapı kütüphaneleri şu anda desteklenmiyor) | 2024 Q2        |
| **Uzun yanıtlarda işbirlikçi konuşan farklı ses modelleri.** | Tamamlandı | 2024 Q2        |
| **Konuşmayı Tamamladığınızda Otomatik Kaydı Durdurma** | Tamamlandı | 2024 Q2        |
| **Uyanma Komutu**                   | Tamamlandı   | 2024 Q2        |
| **Sürekli Konuşmalar**              | Tamamlandı   | 2024 Q2        |
| **Cihazda daha fazla yetenek ekleme** | Planlanıyor | 2024 Q2        |
| DeepFace Entegrasyonu (Yüz Tanıma)  | Planlanıyor  | 2024 Q2        |








## Yetenekler
Şu anda birçok altyapı öğemiz var. ChatGPT uygulamasında zaten bulunan tüm öğeleri sağlamayı hedefliyoruz.

| Yetenek                              | Durum   |
|--------------------------------------|---------|
| **Ekran Okuma**                      | OK      |
| **Ekrandaki Metin veya Simgeye Tıklama** | OK      |
| **Ekrandaki Metin veya Simgeye Taşıma** | OK      |
| **Bir Şeyler Yazma**                 | OK      |
| **Herhangi Bir Tuşa Basma**          | OK      |
| **Kaydırma**                         | OK      |
| **Mikrofon**                         | OK      |
| **Sistem Sesleri**                   | OK      |
| **Bellek**                           | OK      |
| **Uygulama Açma ve Kapatma**         | OK      |
| **Bir URL Açma**                     | OK      |
| **Pano**                             | OK      |
| **Arama Motorları**                  | OK      |
| **Python Yazma ve Çalıştırma**       | OK      |
| **SH (Shell) Yazma ve Çalıştırma**   | OK      |
| **Telegram Hesabınızı Kullanma**     | OK      |
| **Bilgi Yönetimi**                   | OK      |
| **[Daha fazla araç ekle](https://github.com/onuratakan/gpt-computer-assistant/blob/master/gpt_computer_assistant/standard_tools.py)** | ?       |

### Önceden Tanımlı Ajanlar
Eğer etkinleştirirseniz asistanınız bu ekiplerle çalışabilir:

| Takım Adı                               | Durum   |
|-----------------------------------------|---------|
| **search_on_internet_and_report_team**   | OK      |
| **generate_code_with_aim_team_**         | OK      |
| **[Kendi ekleyin](https://github.com/onuratakan/gpt-computer-assistant/blob/master/gpt_computer_assistant/teams.py)** | ?       |



  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/ba590bf8-6059-4cb6-8c4e-6d105ce4edd2" alt="Logo"  >
  </a>




## Katkıda Bulunanlar

<a href="https://github.com/onuratakan/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=onuratakan/gpt-computer-assistant" />
</a>



</p>
