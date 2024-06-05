<p align="center">
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/176c8ddb-219e-444e-8782-1f8c37a92678" alt="Logo" width="250" >
  </a>

  <h3 align="center">GPT Computer Assistant</h3>

  <p align="center">
    gpt-4o for windows, macos and ubuntu
    <br />
   <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki"><strong>Documentation</strong></a>
   .
    <a href="https://github.com/onuratakan/gpt-computer-assistant/#Capabilities"><strong>Explore the capabilities »</strong></a>
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

  </p>
  <p align="center">
  <a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
  </a>
  .
  <img src="https://static.pepy.tech/personalized-badge/gpt-computer-assistant?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20Downloads" alt="pypi_downloads">
  </p>


  <p align="center">
   <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>
   <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/badge/X-Join?style=social&logo=x" width=100></a>

  </p>

|ENGLISH|[简体中文](README.zh_CN.md)|

# GPT Computer Assistant
Hi, this is an alternative work for providing ChatGPT MacOS app to Windows and Linux. In this way this is a fresh and stable work. You can easily install as Python library for this time but we will prepare a pipeline to providing native install scripts (.exe).

Powered by <a href="https://github.com/Upsonic/Tiger"><strong>Upsonic Tiger 🐅</strong></a> An function hub for llm agents.

## Installation && Run
Needed >= Python 3.9
```console
pip3 install gpt-computer-assistant
```

```console
computerassistant
```



### Demo Video (1 min)

https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/26ae3624-e619-44d6-9b04-f39cf1ac1f8f

### Use cases

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






## Roadmap

| Feature                         | Status       | Target Release |
|---------------------------------|--------------|----------------|
| Clear Chat History         | Completed    | Q2 2024        |
| Long Audios Support (Split 20mb)      | Completed    | Q2 2024        |
| Text Inputs               | Completed      | Q2 2024        |
| Just Text Mode (Mute Speech)           | Completed  | Q2 2024        |
| Added profiles (Different Chats)          | Completed    | Q2 2024        |
| More Feedback About Assistant Status                  | Completed    | Q2 2024        |
| **New UI**              | Planned      | Q2 2024        |
| **Our Customizable Agent Infrastructure**              | Planned      | Q2 2024        |
| **Native Applications, exe,dmg,appimage**              | Planned      | Q2 2024        |
| **DeepFace Integration (Facial Recognition)**                    | Planned  | Q2 2024        |
| **Local Mode (With Ollama, Speech, and vision models)**  | Planned  | Q2 2024        |


#### Agent Infrastructure | Coming Soon

```python
from gpt-comptuer-assistant import crew, agent

coder = agent("You are an senior python developer")

manager = agent("You are senior project manager")

assistant = crew(
 [coder, manager]
)

assistant.gui()
```




## Capabilities
At this time we have many infrastructure elements. We just aim to provide whole things that already in ChatGPT app.

| Capability                         | Description                      |
|------------------------------------|----------------------------------|
| **Screen Read**                    |            OK                    |
| **Microphone**                     |            OK                    |
| **System Audio**                  |            OK                    |
| **Memory**                         |            OK                    |
| **Open and Close App**             |            OK                    |
| **Open a URL**                     |            OK                    |
| **Clipboard**                       |            OK                    |
| **Search Engines**                 |            OK                    |
| **Writing and running Python**     |            OK                    |
| **Writing and running SH**    |            OK                    |
| **Using your Telegram Account**    |            OK                    |
| **Knowledge Management**           |            OK                    |






## Usage

![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/20972b1e-6d4f-4314-8470-f2fcf79b6e6d)



** After first click to an option that include microphone or system audio you need to stop with another click to same option.



## Contributors

<a href="https://github.com/onuratakan/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=onuratakan/gpt-computer-assistant" />
</a>
