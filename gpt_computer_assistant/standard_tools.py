from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
import datetime
import traceback

try:
    from .tooler import tool
    from .top_bar_wrapper import wrapper
except:
    from tooler import tool
    from top_bar_wrapper import wrapper

_standard_tools_ = {}


def register_tool(func):
    if func.__name__ not in _standard_tools_:
        _standard_tools_[func.__name__] = tool(func)
    return func


@register_tool
@wrapper
def read_website(url: str, max_content_length: int = 5000) -> dict:
    """
    Read the content of a website and return the title, meta data, content, and sub-links.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        return {"error": f"Failed to retrieve the website content: {e}"}

    soup = BeautifulSoup(html, "html.parser")

    meta_properties = [
        "og:description",
        "og:site_name",
        "og:title",
        "og:type",
        "og:url",
        "description",
        "keywords",
        "author",
    ]
    meta = {}
    for property_name in meta_properties:
        tag = soup.find("meta", property=property_name) or soup.find(
            "meta", attrs={"name": property_name}
        )
        if tag:
            meta[property_name] = tag.get("content", "")

    for ignore_tag in soup(["script", "style"]):
        ignore_tag.decompose()

    title = soup.title.string.strip() if soup.title else ""
    content = soup.body.get_text(separator="\n") if soup.body else ""

    links = []
    for a in soup.find_all("a", href=True):
        link_url = urljoin(url, a["href"])
        links.append({"title": a.text.strip(), "link": link_url})

    content = re.sub(r"[\n\r\t]+", "\n", content)
    content = re.sub(r" +", " ", content)
    content = re.sub(r"[\n ]{3,}", "\n\n", content)
    content = content.strip()

    if len(content) > max_content_length:
        content = content[:max_content_length].rsplit(" ", 1)[0] + "..."

    return {"meta": meta, "title": title, "content": content, "sub_links": links}


@register_tool
@wrapper
def google(query: str, max_number: int = 20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch

        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"


@register_tool
@wrapper
def duckduckgo(query: str, max_number: int = 20) -> list:
    """
    Search the query on DuckDuckGo and return the results.
    """
    try:
        from duckduckgo_search import DDGS

        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"





@register_tool
@wrapper
def open_url(url) -> bool:
    """
    Open the URL in the default web browser.

    :param url: str:
    """
    import webbrowser

    try:
        webbrowser.open(url)
        return True
    except:
        return False



@register_tool
@wrapper
def sleep(seconds: int):
    """
    Sleep for the given number of seconds.
    """
    import time

    time.sleep(seconds)





from langchain_experimental.utilities import PythonREPL

the_py_client = PythonREPL()


@register_tool
@wrapper
def python_repl(code: str) -> str:
    """
    Run and return the given python code in python repl
    """
    return the_py_client.run(code)



@register_tool
@wrapper
def keyboard_write(text: str):
    """
    Write the text using the keyboard (its use pyautogui).
    """
    import pyautogui
    pyautogui.write(text)

@register_tool
@wrapper
def keyboard_press(key: str):
    """
    Press the key using the keyboard (its use pyautogui).
    """
    import pyautogui
    pyautogui.press(key)
    pyautogui.press(key)




@register_tool
@wrapper
def get_current_time() -> str:
    """
    Get the current time in ISO format.
    """
    return datetime.datetime.now().isoformat()


@register_tool
@wrapper
def turn_off_wifi() -> bool:
    """
    Turn off the wifi.
    """
    try:
        from pywifi import ControlPeripheral

        wifi = ControlPeripheral()
        wifi.disable()
        return True
    except:
        return False


@register_tool
@wrapper
def turn_on_wifi() -> bool:
    """
    Turn on the wifi.
    """
    try:
        from pywifi import ControlPeripheral

        wifi = ControlPeripheral()
        wifi.enable()
        return True
    except:
        return False


@register_tool
@wrapper
def connect_wifi(ssid: str, password: str) -> bool:
    """
    Connect to the wifi with the given ssid and password.
    """
    try:
        from pywifi import ControlConnection

        # Arguments passed during object instantiation
        controller = ControlConnection(wifi_ssid=ssid, wifi_password=password)
        controller.wifi_connector()
        return True
    except:
        return False



def get_standard_tools():
    print("Tool len", len(_standard_tools_))
    last_list = [_standard_tools_[each] for each in _standard_tools_]
    return last_list


if __name__ == "__main__":
    print(ask_to_user("What is your age"))
