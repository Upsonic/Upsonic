from langchain.tools import tool

def read_website(url: str, max_content_lenght: int = 5000) -> dict:
    """
    Read the content of a website and return the title, meta data, content and sub links.
    """
    try:
        import requests

        from bs4 import BeautifulSoup
        import re

        html = requests.get(url).text
        soup = BeautifulSoup(html)
        meta_properties = [
            "og:description",
            "og:site_name",
            "og:title",
            "og:type",
            "og:url",
        ]
        meta = {}
        for property_name in meta_properties:
            try:
                tag = soup.find("meta", property=property_name)
                if tag:
                    meta[property_name] = str(tag.get("content", None))
            except AttributeError:
                meta[property_name] = None
        for ignore_tag in soup(["script", "style"]):
            ignore_tag.decompose()
        title = soup.title.string if soup.title else ""
        content = soup.body.get_text() if soup.body else ""
        links = []
        for a in soup.find_all("a", href=True):
            links.append({"title": a.text.strip(), "link": a["href"]})
        content = re.sub(r"[\n\r\t]+", "\n", content)
        content = re.sub(r" +", " ", content)
        content = re.sub(r"[\n ]{3,}", "\n\n", content)
        content = content.strip()
        return {"meta": meta, "title": title, "content": content[:max_content_lenght], "sub_links": links}

    except:
        return "An exception occurred"      
    


def google(query:str, max_number:int=20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch
        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"    



def duckduckgo(query:str, max_number:int=20) -> list:
    """
    Search the query on DuckDuckGo and return the results.
    """
    try:
        from duckduckgo_search import DDGS
        return [result["href"] for result in DDGS().text(query, max_results=max_number)]
    except:
        return "An exception occurred"
    



def copy(text:str):
    """
    Copy the text to the clipboard.
    """
    import pyperclip
    pyperclip.copy(text)




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


def sleep(seconds: int):
    """
    Sleep for the given number of seconds.
    """
    import time
    time.sleep(seconds)


the_standard_tools_ = []


the_standard_tools_.append(tool(read_website))
the_standard_tools_.append(tool(google))
the_standard_tools_.append(tool(duckduckgo))
the_standard_tools_.append(tool(copy))
the_standard_tools_.append(tool(open_url))
the_standard_tools_.append(tool(sleep))


the_standard_tools = the_standard_tools_