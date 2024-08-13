<p align="center">
  <a href="#">
    <img src="https://github.com/user-attachments/assets/df1a3ec1-8c03-43c8-9432-c71358c35b9e" >
  </a>
  <br>

<p align="center">

  <a href="https://docs.gca.dev">
    <img src="https://github.com/user-attachments/assets/c60562bf-540e-47d9-b578-994285071128" width="250">
  </a>
  .
  <a href="https://github.com/Upsonic/gpt-computer-assistant/releases/latest/download/gpt-computer-assistant-openai.dmg">
    <img src="https://github.com/user-attachments/assets/a0475f31-9dfd-4a0c-91b0-7ae128c3c773" width="250">
  </a>
  .
  <a href="https://github.com/Upsonic/gpt-computer-assistant/releases/latest/download/gpt-computer-assistant-openai.exe">
    <img src="https://github.com/user-attachments/assets/c94139fd-609c-4780-9541-6e9e01dd0e47" width="250">
  </a>

</p>


  <p align="center">
    <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>
  </p>
  <p align="center">
    <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/badge/X_App-Join?style=social&logo=x" width=150></a>
  </p>

  <p align="center">
    <br />
    Intelligence development framework
    <br />
    </p>
    <br>

  <p align="center">
  <a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
  </a>
  .
  <img src="https://static.pepy.tech/personalized-badge/gpt-computer-assistant?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20Downloads" alt="pypi_downloads">
  </p>




|ENGLISH|[简体中文](README.zh_CN.md)|[正體中文](README.zh_TW.md)|[TÜRKÇE](README.TR.md)

# GPT Computer Assistant
Hi, this is an alternative work for providing ChatGPT MacOS app to Windows and Linux. In this way this is a fresh and stable work. You can easily install as Python library for this time but we will prepare a pipeline for providing native install scripts (.exe).

Powered by <a href="https://github.com/Upsonic/Tiger"><strong>Upsonic Tiger 🐅</strong></a> A function hub for llm agents.




## Install and run
*Python 3.9 or higher is required

```console
pip install 'gpt-computer-assistant[base]'
pip install 'gpt-computer-assistant[api]'
```

To run gpt-computer-assistant, simply type

```console
computerassistant --api
```


<p align="center">

  <a href="#">
    <img src="https://github.com/user-attachments/assets/890b4e0a-4484-4870-a158-2d365b0d969e" >
  </a>

</p>





### Local text-to-speech | NEW
<details>

Now GCA just support totaly local text-to-speech with Microsoft Open Source model. For enabling and using it you should run this command:

```console
pip3 install 'gpt-computer-assistant[local_tts]'
```

After that, just go to LLM setting section and select `microsoft_local` in tts combobox.
</details>


### Local speech-to-text | NEW
<details>

Now GCA just support totaly local speech-to-text with OpenAI Whisper tiny model. For enabling and using it you should run this commands:

```console
pip3 install 'gpt-computer-assistant[local_stt]'
```

Installing ffmpeg:

```console
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```



After that, just go to LLM setting section and select `openai_whisper_local` in stt combobox.
</details>



### Wake Word
<details>


We have added Pvporcupine integration. To use it, you need to install an additional library:

```console
pip3 install 'gpt-computer-assistant[wakeword]'
```
</details>

### Error Solutions
<details>

#### Setuptools

```console
pip install setuptools --upgrade
```

</details>


After that, please enter your [Pvporcupine](https://picovoice.ai/) API key and enable the wake word feature.
</details>












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





## LLM Settings

```python
from gpt_computer_assistant.remote import remote

remote.save_models("gpt-4o")
remote.save_openai_api_key("sk-**")
```

## Characteristic API

Name of the assitant:
```python
remote.change_name("X Intelligence")
```

Developer personna of the assistant:
```python
remote.change_developer("X Company")
```

## Connect Your Functions API

Installing an library:
```python
remote.install_library("numpy")
```

Adding functianility as python functions:
```python
@remote.custom_tool
def my_server_status() -> bool:
  """
  Check the server status.
  """
  return True
```


## Interact with User API


### remote.input
Talk with assistant, about user and computer. With this api you can create an consulting process.

`Hi, look to user window and return which app using now`

`Ask user to is user need any kind of supoprt`

`Extract the user redis config file.`


With this questions you will make a shortcut for your needs. 
**You can collect informations from user computer or directly from user or user computer.**

```python
output = remote.input("Extract the user redis config file.", screen=False)
print(output)
```


### remote.operation

With this you can define an operation and inform user via GUI


```python
with remote.operation("Scanning"):
  remote.wait(5)
```




## remote.say

This function say given string to user via `text-to-speech`

```python
remote.say("Hello, sir how are you today?")
```




## remote.ask

This for getting an user answer for any question.


```python
the_user_answer = remote.ask("What is your favorite color?")
print(the_user_answer)
```


## remote.screenshot_to_memory

This function is so important, if you want to take a cold screenshot to assistant history to remember and taking some details in future, this function for you.


```python
remote.screenshot_to_memory()
```



## Usage
![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/37d34745-ae4b-4b37-9bfa-aec070c97897)



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
