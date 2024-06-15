

<p align="center">
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/176c8ddb-219e-444e-8782-1f8c37a92678" alt="Logo" width="250" >
  </a>

  <h3 align="center">GPT 電腦助手</h3>
  <p align="center">
  <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/discord/1148697961639968859.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2" width=100></a>
  </p>

  <p align="center">
    適用於 Windows、MacOS 和 Ubuntu 的 gpt-4o
    <br />
   <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki"><strong>文件</strong></a>
   .
    <a href="https://github.com/onuratakan/gpt-computer-assistant/#Capabilities"><strong>探索功能 »</strong></a>
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
   <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/twitter/follow/GPTCompAsst?style=social" width=160></a>
</p>


|[ENGLISH](README.md)|[簡體中文](README.zh_CN.md)|正體中文

# GPT 電腦助手
嗨，這是為了將 ChatGPT MacOS 應用程式提供給 Windows 和 Linux 的替代方案。這樣做可以提供一個新鮮且穩定的解決方案。這次您可以輕鬆地安裝為 Python 庫，但我們將準備一個流程，提供本機安裝腳本（.exe）。

由 <a href="https://github.com/Upsonic/Tiger"><strong>Upsonic Tiger 🐅</strong></a> 提供支持的功能集成中心。

 <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki/Usage"><img alt="Static Badge" src="https://img.shields.io/badge/Local_Models-Available-blue" width=150></a>
 <br>
 <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki/Usage"><img alt="Static Badge" src="https://img.shields.io/badge/Groq-Available-blue" width=100></a>



## 安裝 && 運行
需要 >= Python 3.9
```console
pip3 install 'gpt-computer-assistant[default]'
```

```console
computerassistant
```

### 代理基礎設施

這樣一來，您可以創建 `crewai` 代理，並將其用於 gpt-computer-assistant 圖形用戶界面和工具中。


```console
pip3 install 'gpt-computer-assistant[agentic]'
```

```python
from gpt_computer_assistant import Agent, start

manager = Agent(
  role='Project Manager',
  goal='understands project needs and assist coder',
  backstory="""You're a manager at a large company.""",
)

coder = Agent(
  role='Senior Python Coder',
  goal='writing python scripts and copying to clipboard',
  backstory="""You're a python developer at a large company.""",
)


start()
```


### 新增自訂工具

現在您可以添加在代理基礎設施和助理進程中運行的自訂工具。


```python
from gpt_computer_assistant import Tool, start

@Tool
def sum_tool(first_number: int, second_number: int) -> str:
    """Useful for when you need to sum two numbers together."""
    return first_number + second_number

start()
```


https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/26ae3624-e619-44d6-9b04-f39cf1ac1f8f

<p align="center">
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/94ac619c-1f29-4fe6-b3cb-85a03932646b" alt="Logo"  >
  </a>
</p>







## 使用方式
![選項](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/54b39347-98e0-4ee4-a715-9128c40dbcd4)


## 使用案例

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






## 路線圖

| 功能                             | 狀態         | 目標發布      |
|---------------------------------|--------------|--------------|
| 清除聊天記錄                     | 已完成       | 2024 年第二季度|
| 長音訊支持（拆分 20mb）          | 已完成       | 2024 年第二季度|
| 文本輸入                          | 已完成       | 2024 年第二季度|
| 僅文本模式（靜音）                | 已完成       | 2024 年第二季度|
| 添加配置文件（不同聊天）           | 已完成       | 2024 年第二季度|
| 更多關於助手狀態的回饋             | 已完成       | 2024 年第二季度|
| **新 UI**                        | 計劃中       | 2024 年第二季度|
| **我們的自訂代理基礎設施**        | 計劃中       | 2024 年第二季度|
| **本機應用程式，exe，dmg，appimage** | 計劃中       | 2024 年第二季度|
| **DeepFace 集成（臉部識別）**      | 計劃中       | 2024 年第二季度|
| **本地模式（使用 Ollama，語音和視覺模型）** | 計劃中       | 2024 年第二季度|








## 功能
此時我們擁有許多基礎設施元素。我們只是希望提供 ChatGPT 應用中已經存在的所有功能。

| 功能                              | 描述                          |
|-----------------------------------|-------------------------------|
| **螢幕讀取**                      |            OK                 |
| **麥克風**                        |            OK                 |
| **系統音訊**                      |            OK                 |
| **記憶體**                          |            OK                 |
| **打開和關閉應用程式**              |            OK                 |
| **打開一個 URL**                  |            OK                 |
| **剪貼簿**                        |            OK                 |
| **搜尋引擎**                      |            OK                 |
| **編寫和運行 Python**            |            OK                 |
| **編寫和運行 SH**                |            OK                 |
| **使用你的 Telegram 帳戶**        |            OK                 |
| **知識管理**                      |            OK                 |










## 貢獻者

<a href="https://github.com/onuratakan/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=onuratakan/gpt-computer-assistant" />
</a>
