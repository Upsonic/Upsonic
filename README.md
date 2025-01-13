

  
</p>

## What is GCA?

Hi, this is an open source framework to build vertical AI agent. We just support many llms and new technologies like mcp. You can build your own vertical ai agent army in few commands with the stucturized API.


<p>



  <p >
    <a href="https://discord.gg/qApFmWMt8x"><img alt="Static Badge" src="https://img.shields.io/badge/Discord-Join?style=social&logo=discord" width=200></a>
    .
    <a href="https://x.com/GPTCompAsst"><img alt="Static Badge" src="https://img.shields.io/badge/X_App-Join?style=social&logo=x" width=160></a>
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


## Using GCA.dev Cloud

<b>Installation</b>
```console
pip install gpt-computer-assistant
```

Single Instance:
```python
from gpt_computer_assistant import Cloud, Task, TypeVerifier

# Starting instance
agent = Cloud.agent()


# Run task
star_number = agent.run(
    Task(
        "Extract the github star number of https://github.com/Upsonic/gpt-computer-assistant", 
        TypeVerifier("integer")
    )
)
print(star_number)


agent.close()
```



<p align="center">
<br>
<br>
<br>
</p>


## Self-Hosted GCA Server

### Docker

**Pulling Image**

* If you are using ARM computer like M Chipset macbooks you should use *ARM64* at the end.

```console
docker pull upsonic/gca_docker_ubuntu:dev0-AMD64
```

**Starting container**

```console
docker run -d -p 5901:5901 -p 7541:7541 upsonic/gca_docker_ubuntu:dev0-AMD64
```

**LLM Settings&Using**

```python
from gpt_computer_assistant import docker

# Starting instance
agent = docker.agent("http://localhost:7541/")

# Connecting to OpenAI and Anthropic
agent.client.save_model("gpt-4o")
agent.client.save_openai_api_key("sk-**")
agent.client.save_anthropic_api_key("sk-**")

# Asking and getting result
result = agent.request("Extract the tech stacks of gpt-computer-assitant Company", "i want a list")
print(result)

agent.close()
```


<p align="center">
<br>
<br>
</p>


### Local
<b>Installation</b>
```console
pip install 'gpt-computer-assistant[base]'
pip install 'gpt-computer-assistant[api]'
```

<b>LLM Settings&Using</b>

```python
from gpt_computer_assistant import local

# Starting agent
agent = local.agent()

# Connecting to OpenAI and Anthropic
agent.client.save_model("gpt-4o")
agent.client.save_openai_api_key("sk-**")
agent.client.save_anthropic_api_key("sk-**")

# Asking and getting result
result = agent.request("Extract the tech stacks of gpt-computer-assitant Company", "i want a list")
print(result)

agent.close()
```

<p align="center">
<br>
  <br>
  <br>
</p>



## Adding Custom MCP Server to GCA

```python
instance.client.add_mcp_server("websearch", "npx", ["-y", "@mzxrai/mcp-webresearch"])
```


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

<a href="https://github.com/upsonic/gpt-computer-assistant/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=upsonic/gpt-computer-assistant" />
</a>
