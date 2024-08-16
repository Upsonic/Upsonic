import requests
from bs4 import BeautifulSoup

try:
    from ..character import set_website_content
except:
    from gpt_computer_assistant.character import set_website_content


def train(url: str) -> bool:
    try:
        # Go to url and extract these elements
        meta_properties = [
            "og:description",
            "og:site_name",
            "og:title",
            "og:type",
            "og:url",
        ]

        # Fetch the webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the meta tags
        meta_tags = soup.find_all("meta")

        # Initialize the data dictionary
        data = {}

        # Loop through the meta tags and extract the content
        for tag in meta_tags:
            if tag.get("property") in meta_properties:
                data[tag.get("property")] = tag.get("content")

        # Also add the other useful information texts from the webpage
        data["title"] = soup.title.string
        data["h1"] = soup.h1.string
        data["p"] = soup.p.string

        text = soup.get_text(separator="\n", strip=True)

        data["text"] = text

        data["url"] = url

        # Now create an string with good looking like this
        # Title: {title}

        the_string = ""

        for key, value in data.items():
            the_string += f"{key}: {value}\n"

        set_website_content(the_string)

        return True

    except Exception as e:
        return e
