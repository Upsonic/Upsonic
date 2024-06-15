<p align="center">
  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/176c8ddb-219e-444e-8782-1f8c37a92678" alt="Logo" width="250" >
  </a>

  <h3 align="center">GPT 计算机助手</h3>

  <p align="center">
    适用于 Windows、MacOS 和 Ubuntu 的 gpt-4o
    <br />
   <a href="https://github.com/onuratakan/gpt-computer-assistant/wiki"><strong>文档</strong></a>
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
   <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>
   <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/badge/X-Join?style=social&logo=x" width=100></a>

  </p>

|[ENGLISH](README.md)|简体中文|[正體中文](README.zh_TW.md)

# GPT 计算机助手
你好，这是一个将 ChatGPT MacOS 应用程序提供给 Windows 和 Linux 的替代工作。因此，这是一个全新且稳定的项目。此时，您可以轻松地将其作为 Python 库安装，但我们将准备一个流水线来提供本机安装脚本 (.exe)。

由 <a href="https://github.com/Upsonic/Tiger"><strong>Upsonic Tiger 🐅</strong></a> 提供支持的功能集成中心。

## 安装 && 运行
需要 >= Python 3.9
```console
pip3 install 'gpt-computer-assistant[default]'
```

```console
computerassistant
```



### 演示视频（1 分钟）

https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/26ae3624-e619-44d6-9b04-f39cf1ac1f8f



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



## 路线图

| 功能                             | 状态         | 目标发布      |
|---------------------------------|--------------|--------------|
| 清除聊天记录                     | 已完成       | 2024 年第二季度|
| 长音频支持（拆分 20mb）          | 已完成       | 2024 年第二季度|
| 文本输入                          | 已完成       | 2024 年第二季度|
| 仅文本模式（静音）                | 已完成       | 2024 年第二季度|
| 添加配置文件（不同聊天）           | 已完成       | 2024 年第二季度|
| 更多关于助手状态的反馈             | 已完成       | 2024 年第二季度|
| **新 UI**                        | 计划中       | 2024 年第二季度|
| **我们的自定义代理基础设施**        | 计划中       | 2024 年第二季度|
| **本机应用程序，exe，dmg，appimage** | 计划中       | 2024 年第二季度|
| **DeepFace 集成（面部识别）**      | 计划中       | 2024 年第二季度|
| **本地模式（使用 Ollama，语音和视觉模型）** | 计划中       | 2024 年第二季度|


#### 代理基础设施 | 即将推出

```python
from gpt-comptuer-assistant import crew, agent

coder = agent("你是一名高级 Python 开发者")

manager = agent("你是一名高级项目经理")

assistant = crew(
 [coder, manager]
)

assistant.gui()
```




## 功能
此时我们拥有许多基础设施元素。我们只是希望提供 ChatGPT 应用中已经存在的所有功能。

| 功能                              | 描述                          |
|-----------------------------------|-------------------------------|
| **屏幕读取**                      |            OK                 |
| **麦克风**                        |            OK                 |
| **系统音频**                      |            OK                 |
| **内存**                          |            OK                 |
| **打开和关闭应用程序**              |            OK                 |
| **打开一个 URL**                  |            OK                 |
| **剪贴板**                        |            OK                 |
| **搜索引擎**                      |            OK                 |
| **编写和运行 Python**            |            OK                 |
| **编写和运行 SH**                |            OK                 |
| **使用你的 Telegram 账户**        |            OK                 |
| **知识管理**                      |            OK                 |







## 用法

![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/20972b1e-6d4f-4314-8470-f2fcf79b6e6d)



** 第一次单击包含麦克风或系统音频的选项后，需要再次单击相同选项以停止。



## 贡献者

<a href="https://github.com/onuratakan/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=onuratakan/gpt-computer-assistant" />
</a>