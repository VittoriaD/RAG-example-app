import requests
from bs4 import BeautifulSoup, Tag
import os


def get_images_by_section(soup):
    article = soup.find('article')
    target_figs = article.find_all('figure')

    images_by_title = {}

    for element in target_figs:
        src = element.img.get('src')
        next_el = element.next_sibling
        if next_el and next_el.name == 'h1':
            images_by_title[next_el.text.strip()] = [src]

    return images_by_title


class TheBatchScraper:
    BASE_URL = "https://www.deeplearning.ai/the-batch/"
    ROOT_URL = "https://www.deeplearning.ai/"

    def __init__(self):
        self.session = requests.Session()

    def get_article_links(self, limit=None):
        response = self.session.get(self.BASE_URL)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('article', limit=limit)
        hrefs = []

        for article in articles:
            a_tag = article.find('a', href=lambda x: x and x.startswith('/the-batch/issue-'))
            if a_tag:
                full_url = self.ROOT_URL + a_tag['href']
                hrefs.append(full_url)

        return hrefs

    def get_article_content(self, url):
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_name = soup.find('h1').text.strip()
        article = soup.find('article')
        sub_articles = article.find_all('h1')[1:]

        content = []

        for sub_article in sub_articles:
            name = sub_article.text.strip()
            paragraphs = []

            next_sibling = sub_article.next_sibling
            while next_sibling and next_sibling.name != 'h1':
                if next_sibling.name != 'figure' and next_sibling.text:
                    paragraphs.append(next_sibling.text.strip())
                next_sibling = next_sibling.next_sibling

            content.append({
                'title': name,
                'paragraphs':  paragraphs,
                'images': []
            })

        return {'page_name': page_name, 'content': content, 'soup': soup}

    def scrape_latest(self, limit=3):
        results = []
        for url in self.get_article_links(limit=limit):
            print(f"Scraping: {url}")
            article = self.get_article_content(url)
            images_by_section = get_images_by_section(article["soup"])

            for section in article["content"]:
                section_title = section["title"]
                section["images"] = images_by_section.get(section_title, [])

            results.append({
                "page_name": article["page_name"],
                "content": article["content"]
            })
        return results
