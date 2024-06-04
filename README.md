# GPT Computer Assistant
Hi, this is an alternative work for providing ChatGPT MacOS app to Windows and Linux. In this way this is a fresh and stable work. You can easily install as Python library for this time but we will prepare a pipeline to providing native install scripts (.exe).

Powered by [Upsonic Tiger 🐅](https://github.com/Upsonic/Tiger) An function hub for llm agents.



<a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>

## Demo Video (1 min)



https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/082a4ee7-e09f-4426-9059-e49b353976a0


## Installation && Run
Needed >= Python 3.9
```console
pip3 install gpt-computer-assistant
```

```console
computerassistant
```

Profile Selection, different profiles have fully different message histories. Just like: work, personal, upsonic.

```python
computerassistant --profile personal
```


### Upgrade
```python
pip3 install gpt-computer-assistant --upgrade
```




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
| **Local Mode (With Ollama, Speech, and vision models)**  | Planned  | Q4 2024        |
| **Writing and Running Scripts** | Planned      | Q2 2024        |
| **Using your Telegram Account** | Planned      | Q3 2024        |
| **Knowledge Management**        | Planned  | Q1 2024        |


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




## Use cases

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


## Usage
![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/91fe041d-4a3f-4a6e-9294-20ce2fa1ca36)


** After first click to an option that include microphone or system audio you need to stop with another click to same option.


### About Macos
For now we have some problems with **intel**, but we will solve it. 

https://github.com/onuratakan/gpt-computer-assistant/issues/10
