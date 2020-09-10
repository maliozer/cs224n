from bs4 import BeautifulSoup
from bs4.element import Comment
import requests


class PageScraper:
    def __init__(self):
        pass

    def tag_visible(self,element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self,body):
        soup = BeautifulSoup(body, 'html.parser', from_encoding='utf8')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

    def req(self,url):
        r = requests.get(url)
        return r
    
    def get_page_text(self, url):
        res = self.req(url)
        
        #debug message
        print(res.status_code, res.encoding)

        text = self.text_from_html(res.content)
        return text