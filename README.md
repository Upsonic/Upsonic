<p align="center">
  <a href="#">
    <img src="https://github.com/user-attachments/assets/df1a3ec1-8c03-43c8-9432-c71358c35b9e" >
  </a>
  <br>

<p align="center">

  <a href="https://docs.gca.dev">
    <img src="https://github.com/user-attachments/assets/c60562bf-540e-47d9-b578-994285071128" width="250">
  </a>

</p>

  <p align="center">
    <a href="https://www.producthunt.com/posts/gpt-computer-assistant?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-gpt&#0045;computer&#0045;assistant" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=465468&theme=dark&period=daily" alt="GPT&#0032;Computer&#0032;Assistant - Create&#0032;intelligence&#0032;for&#0032;your&#0032;products | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
    .
    <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=150></a>
    .
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

# GPT Computer Assistant(GCA)
Hi, this is an alternative work for providing ChatGPT MacOS app to Windows and Linux. In this way this is a fresh and stable work. You can easily install as Python library for this time but we will prepare a pipeline for providing native install scripts (.exe).

Powered by <a href="https://github.com/Upsonic/Tiger"><strong>Upsonic Tiger 🐅</strong></a> A function hub for llm agents.




## 1. Install and run
**Python 3.10 or 3.11 is required**

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




<p align="center">
<br>
  <br>
  <br>
  <br>
  <br>
</p>






## 2. LLM Settings

```python
from gpt_computer_assistant.remote import remote

remote.save_models("gpt-4o")
remote.save_openai_api_key("sk-**")
```

<p align="start">

  <a href="https://docs.upsonic.co/gca/dev_guides/llm_settings">
    <img src="https://github.com/user-attachments/assets/a75c8ddf-f9df-436b-9dc8-c5220211e15e" width="150">
  </a>

</p>



<p align="center">
<br>
  <br>
  <br>
</p>



## 3. Characteristic API


```python
# Name of the assitant:
remote.change_name("X Intelligence")

#Developer personna of the assistant:
remote.change_developer("X Company")
```

<p align="start">

  <a href="https://docs.upsonic.co/gca/dev_guides/characteristic">
    <img src="https://github.com/user-attachments/assets/d7e02ac6-e40c-4b35-8e65-4621bf3fb9a1" width="150">
  </a>

</p>



<p align="center">
<br>
  <br>
  <br>
</p>



## 4. Connect Your Functions API


```python
# Installing an library:
remote.install_library("numpy")



# Adding functianility as python functions:
@remote.custom_tool
def my_server_status() -> bool:
  """
  Check the server status.
  """
  return True
```



<p align="center">
<br>
  <br>
  <br>
</p>



## 5. Interact with User API


### remote.input

Talk with assistant, about user and computer. With this api you can create an consulting process.

```markdown
`Hi, look to user window and return which app using now`

`Ask user to is user need any kind of supoprt`

`Extract the user redis config file.`
```

With this questions you will make a shortcut for your needs. 
**You can collect informations from user computer or directly from user or user computer.**

```python
output = remote.input("Extract the user redis config file.", screen=False)
print(output)
```


<p align="start">

  <a href="https://docs.upsonic.co/gca/dev_guides/interact">
    <img src="https://github.com/user-attachments/assets/81614347-ab85-4965-9b77-225d0f2961e9" width="150">
  </a>
  .
  <a href="https://docs.upsonic.co/gca/dev_guides/interact">
    <img src="https://github.com/user-attachments/assets/ecaa7590-f4c5-4eda-9482-462cef54aeff" width="150">
  </a>
  .
  <a href="https://docs.upsonic.co/gca/dev_guides/interact">
    <img src="https://github.com/user-attachments/assets/0f35df10-b32e-4fa1-936e-b336be46b1bd" width="150">
  </a>

</p>



<p align="start">

  <a href="https://docs.upsonic.co/gca/dev_guides/interact">
    <img src="https://github.com/user-attachments/assets/a893c50c-3ede-4b42-90ee-92e2fea82120" width="150">
  </a>

</p>


<p align="center">
<br>
  <br>
  <br>
</p>


## Usage
![options](https://github.com/onuratakan/gpt-computer-assistant/assets/41792982/37d34745-ae4b-4b37-9bfa-aec070c97897)



### Use cases


<img alt="Screenshot 2024-08-13 at 18 33 52" src="https://github.com/user-attachments/assets/8f994160-893a-4f56-bbf0-4a7aa87af650">




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
