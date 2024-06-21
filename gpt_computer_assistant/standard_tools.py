from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin

from .tooler import tool



from .top_bar_wrapper import wrapper


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
def google(query:str, max_number:int=20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch
        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"    


@wrapper
def duckduckgo(query:str, max_number:int=20) -> list:
    """
    Search the query on DuckDuckGo and return the results.
    """
    try:
        from duckduckgo_search import DDGS
        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"
    


@wrapper
def copy(text:str):
    """
    Copy the text to the clipboard.
    """
    import pyperclip
    pyperclip.copy(text)



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

@wrapper
def sleep(seconds: int):
    """
    Sleep for the given number of seconds.
    """
    import time
    time.sleep(seconds)




@wrapper
def keyboard_write(text:str):
    """
    Write the text using the keyboard.
    """
    import pyautogui
    pyautogui.write(text)



@wrapper
def keyboard_press(key:str):
    """
    Press the key using the keyboard.
    """
    import pyautogui
    pyautogui.press(key)







def get_standard_tools():

    the_standard_tools_ = []


    the_standard_tools_.append(tool(read_website))
    the_standard_tools_.append(tool(google))
    the_standard_tools_.append(tool(duckduckgo))
    the_standard_tools_.append(tool(copy))
    the_standard_tools_.append(tool(open_url))
    the_standard_tools_.append(tool(sleep))
    the_standard_tools_.append(tool(keyboard_write))
    the_standard_tools_.append(tool(keyboard_press))


    the_standard_tools = the_standard_tools_

    return the_standard_tools