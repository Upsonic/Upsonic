from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
from urllib.parse import urljoin

from .tooler import tool
from .top_bar_wrapper import wrapper

_standard_tools_ = []

def register_tool(func):
    _standard_tools_.append(tool(func))
    return func
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
        "author"
    ]
    meta = {}
    for property_name in meta_properties:
        tag = soup.find("meta", property=property_name) or soup.find("meta", attrs={"name": property_name})
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
        content = content[:max_content_length].rsplit(' ', 1)[0] + '...'

    return {"meta": meta, "title": title, "content": content, "sub_links": links}



@wrapper
@register_tool
def google(query: str, max_number: int = 20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch
        return list(gsearch(query, stop=max_number))
    except ImportError:
        return {"error": "Google search library not found."}
        from googlesearch import search as gsearch
        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"    


@wrapper
@register_tool
def duckduckgo(query: str, max_number: int = 20) -> list:
    """
    Search the query on DuckDuckGo and return the results.
    """
    try:
        from duckduckgo_search import DDGS
        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"
        from duckduckgo_search import DDGS
        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"



@wrapper
@register_tool
def copy(text: str):
    """
    Copy the text to the clipboard.
    """
    import pyperclip
    pyperclip.copy(text)
    pyperclip.copy(text)



@wrapper
@register_tool
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
        return False

@wrapper
@register_tool
def sleep(seconds: int):
    """
    Sleep for the given number of seconds.
    """
    import time
    time.sleep(seconds)




@wrapper
@register_tool
def keyboard_write(text: str):
    """
    Write the text using the keyboard.
    """
    import pyautogui
    pyautogui.write(text)


@wrapper
@register_tool
def keyboard_press(key: str):
    """
    Press the key using the keyboard.
    """
    import pyautogui
    pyautogui.press(key)
    pyautogui.press(key)



from langchain_experimental.utilities import PythonREPL

the_py_client = PythonREPL()

@wrapper
@register_tool
def python_repl(code: str) -> str:
    """
    Run and return the given python code in python repl
    """
    return the_py_client.run(code)


@wrapper
@register_tool
def app_open(app_name: str) -> bool:
    """
    Opens the native apps.
    """
    try:
        from AppOpener import open
        open(app_name, throw_error=True)
        return True
    except:
        try:
            from MacAppOpener import open
            open(app_name)
        except:
            return False

@wrapper
@register_tool
def app_close(app_name: str) -> bool:
    """
    Closes the native apps.
    """
    try:
        from AppOpener import close
        close(app_name, throw_error=True)
        return True
    except:
        try:
            from MacAppOpener import open
            close(app_name)
        except:
            return False
        from AppOpener import close
        close(app_name, throw_error=True)
        return True
    except:
        try:
            from MacAppOpener import open
            close(app_name)
        except:
            return False


def get_standard_tools():
    return _standard_tools_
