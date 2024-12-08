<p align="center">
  <a href="#">
    <img src="https://github.com/user-attachments/assets/27778034-29f5-4a71-b696-4e3f70760b26" >
  </a>
</p>

## What is GCA?

Hi, this is an open source framework to build vertical AI agent. We just support many llms and new technologies like mcp. You can build your own vertical ai agent army in few commands with the stucturized API.


<p>





  <p >
    <a href="https://www.producthunt.com/posts/gpt-computer-assistant?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-gpt&#0045;computer&#0045;assistant" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=465468&theme=dark&period=daily" alt="GPT&#0032;Computer&#0032;Assistant - Create&#0032;intelligence&#0032;for&#0032;your&#0032;products | Product Hunt" width="200"  /></a>
    .
    <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=120></a>
    .
    <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/badge/X_App-Join?style=social&logo=x" width=100></a>
  </p>




  <p>
  <a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
  </a>
  .
  <img src="https://static.pepy.tech/personalized-badge/gpt-computer-assistant?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20Downloads" alt="pypi_downloads">
  </p>





<p align="center">
<br>

  <br>
</p>


# Playground of GCA | NEW

With [playground.gca.dev](https://playground.gca.dev/) you are ready to test and create your own strategies for creating an Vertical AI Agent.

- Playground sessions limited to <b>10 minute</b>.

<a href="https://playground.gca.dev/">
  <img src="https://github.com/user-attachments/assets/6eb98bc0-5050-4a78-93f9-6913ef7a23b0" alt="Playground"  width=1000>
</a>

<p align="center">
<br>
  <br>

</p>

# GPT Computer Assistant(GCA)
GCA is an AI agent framework designed to make computer use across Windows, macOS, and Ubuntu. GCA enables you to replace repetitive, small-logic-based tasks worker to an AI. There is an really important potential that we believe. Whether you’re a developer, analyst, or IT professional, GCA can empower you to accomplish more in less time.


Imagine this:



- <b>Extract the tech stacks of xxx Company</b>             | Sales Development Representer
- <b>Identify Relevant tables for Analysis for xxx</b>      | Data Analytics
- <b>Check the logs to find core cause of this incident</b> | Technical Support Engineer
- <b>Making CloudFlare Security Settings</b>                | Security Specialist


These examples shows how GCA is realize the concept of <b>Vertical AI Agents</b> solutions that not only replicate human tasks, GCA also in the beyond of human speed at same cases.


<p align="center">
<br>
  <br>

</p>


## How GCA Works?


GCA is a Python-based project that runs on multiple operating systems, including Windows, macOS, and Ubuntu. It integrates external concepts, like the Model Context Protocol (MCP), along with its own modules, to interact with and control a computer efficiently. The system performs both routine and advanced tasks by mimicking human-like actions and applying computational precision.



### 1.	Human-like Actions:
GCA can replicate common user actions, such as:
-	<b>Clicking</b>: Interact with buttons or other UI elements.
-	<b>Reading</b>: Recognize and interpret text on the screen.
-	<b>Scrolling</b>: Navigate through documents or web pages.
-	<b>Typing</b>: Enter text into forms or other input fields.
### 2.	Advanced Capabilities:
Through MCP and GCA’s own modules, it achieves tasks that go beyond standard human interaction, such as:

-	<b>Updating dependencies</b> of a project in seconds.
-	<b>Analyzing entire database</b> tables to locate specific data almost instantly.
- <b>Automating cloud security</b> configurations with minimal input.




<p align="center">
<br>
  <br>
  <br>

</p>


## Prequisites
- Python 3.10

<p align="center">
<br>
  <br>

</p>

## Using GCA.dev Cloud

<b>Installation</b>
```console
pip install gpt-computer-assistant
```

Single Instance:
```python
from gpt_computer_assistant import cloud

# Starting instance
instance = cloud.instance()

# Show Screenshot
instance.current_screenshot()

# Asking and getting result
result = instance.request("Extract the tech stacks of gpt-computer-assitant Company", "i want a list")
print(result)


instance.close()
```

<img src="https://github.com/user-attachments/assets/3fd70530-6b86-43b4-9025-dce7853e4a38" alt="Cloud"  width=1000>




<p align="center">
<br>

</p>


## Self-Hosted GCA Server

### Local
<b>Installation</b>
```console
pip install 'gpt-computer-assistant[base]'
pip install 'gpt-computer-assistant[api]'
```

<b>LLM Settings</b>

```python
from gpt_computer_assistant import local

# Starting instance
instance = local.instance()

# Connecting to OpenAI
instance.client.save_models("gpt-4o")
instance.client.save_openai_api_key("sk-**")

# Asking and getting result
result = instance.request("Extract the tech stacks of gpt-computer-assitant Company", "i want a list")
print(result)

instance.close()
```


<img width="1000" src="https://github.com/user-attachments/assets/327cdceb-49e7-4a8a-a724-e386553f43d8">


<p align="center">
<br>
  <br>
  <br>
</p>







## Roadmap
| Feature                         | Status       | Target Release |
|---------------------------------|--------------|----------------|
| Clear Chat History         | Completed    | Q2 2024        |
| Long Audios Support (Split 20mb)      | Completed    | Q2 2024        |
| Text Inputs               | Completed      | Q2 2024        |
| Just Text Mode (Mute Speech)           | Completed  | Q2 2024        |
| Added profiles (Different Chats)          | Completed    | Q2 2024        |
| More Feedback About Assistant Status                  | Completed    | Q2 2024        |
| Local Model Vision and Text (With Ollama, and vision models)  | Completed  | Q2 2024        |
| **Our Customizable Agent Infrastructure**              | Completed      | Q2 2024        |
| Supporting Groq Models  | Completed  | Q2 2024        |
| **Adding Custom Tools**  | Completed  | Q2 2024        |
| Click on something on the screen (text and icon)              | Completed      | Q2 2024        |
| New UI              | Completed      | Q2 2024        |
| Native Applications, exe, dmg              | Completed     | Q3 2024        |
| **Collaborated Speaking Different Voice Models on long responses.**              | Completed     | Q2 2024        |
| **Auto Stop Recording, when you complate talking**              | Completed     | Q2 2024        |
| **Wakeup Word**              | Completed     | Q2 2024        |
| **Continuously Conversations**              | Completed     | Q2 2024        |
| **Adding more capability on device**              | Completed     | Q2 2024        |
| **Local TTS**              | Completed     | Q3 2024        |
| **Local STT**              | Completed     | Q3 2024        |
| Tray Menu              | Completed     | Q3 2024        |
| New Line (Shift + Enter)              | Completed     | Q4 2024        |
| Copy Pasting Text Compatibility            | Completed     | Q4 2024        |
| **Global Hotkey**              | On the way     | Q3 2024        |
| DeepFace Integration (Facial Recognition)                    | Planned  | Q3 2024        |







## Capabilities
At this time we have many infrastructure elements. We just aim to provide whole things that already in ChatGPT app.

| Capability                         | Status                      |
|------------------------------------|----------------------------------|
| **Local LLM with Vision (Ollama)**                    |            OK                    |
| Local text-to-speech                    |            OK                    |
| Local speech-to-text                    |            OK                    |
| **Screen Read**                    |            OK                    |
| **Click to and Text or Icon in the screen**                    |            OK                    |
| **Move to and Text or Icon in the screen**                    |            OK                    |
| **Typing Something**                    |            OK                    |
| **Pressing to Any Key**                    |            OK                    |
| **Scrolling**                    |            OK                    |
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
| **[Add more tool](https://github.com/onuratakan/gpt-computer-assistant/blob/master/gpt_computer_assistant/standard_tools.py)**           |            ?                    |

### Predefined Agents
If you enable it your assistant will work with these teams:

| Team Name                         | Status                      |
|------------------------------------|----------------------------------|
| **search_on_internet_and_report_team**                    |            OK                    |
| **generate_code_with_aim_team_**                    |            OK                    |
| **[Add your own one](https://github.com/onuratakan/gpt-computer-assistant/blob/master/gpt_computer_assistant/teams.py)**                    |            ?                    |



  <a href="#">
    <img src="https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/ba590bf8-6059-4cb6-8c4e-6d105ce4edd2" alt="Logo"  >
  </a>




## Contributors

<a href="https://github.com/onuratakan/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=onuratakan/gpt-computer-assistant" />
</a>
