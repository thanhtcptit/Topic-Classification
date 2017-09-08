from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from xvfbwrapper import Xvfb
import re
from utils import *

# Tìm tất cả xâu con trong xâu
def find_all_ss(patern, string):
    return [m.start() for m in re.finditer(patern, string)]

# Tìm vị trí ký tự kết thúc
def end_index(from_idx, string):
    end_idx = '\"'
    for i in range(from_idx + 1, len(string)):
        if string[i] == end_idx:
            return i

# Lấy link của từng topic
def get_topic_links(url, topics):
    driver.get(url)
    html = driver.page_source
    soup = bs(html, 'lxml')
    urls = []
    for t in topics:
        string = soup.find_all(title=t)[0]
        urls.append(string['href'])
    return urls

# Lấy link của từng bài viết trong topic
def get_article_fixed(topic_urls, topic_name, btn_name, attrs, class_name, page_count, fixed_link=False):
    file = open(HTML_LINK_DIR + topic_name, 'w')
    fixed = ''
    if fixed_link:
        fixed = topics_urls[topics.index(topic_name)]
    for url in topic_urls:
        driver.get(url)
        driver.implicitly_wait(5)
        # Request đến nút 'Xem thêm' để lấy link bài viết
        for j in range(page_count):
            driver.find_element_by_class_name(btn_name).click()
        html = driver.page_source
        s = bs(html, 'lxml')
        soup = str(s.find_all(attrs, {'class' : class_name})[0])
        start_idx = find_all_ss('href', soup)
        for i in start_idx:
            j = end_index(i + 6, soup)
            file.write(fixed + soup[i + 6:j] + '\n')
    file.close()

if __name__ == '__main__':
    topics = ['Giáo dục', 'Chính trị - Xã hội', 'Pháp luật', 'Kinh tế',
                                 'Văn hóa - Giải trí', 'Sống khỏe', 'Nhịp sống trẻ', 'Nhịp sống số', 'Thể thao', 'Du lịch']

    sports_links = ['http://thethao.tuoitre.vn/tin/bong-da', 'http://thethao.tuoitre.vn/tin/quan-vot'
                        , 'http://thethao.tuoitre.vn/tin/cac-mon-khac']
    tech_links = ['http://nhipsongso.tuoitre.vn/dien-thoai.htm', 'http://nhipsongso.tuoitre.vn/thiet-bi-so.htm',
                  'http://nhipsongso.tuoitre.vn/bao-mat.htm', 'http://nhipsongso.tuoitre.vn/tu-van-tieu-dung.htm'
                  ,'http://nhipsongso.tuoitre.vn/blog.htm', 'http://nhipsongso.tuoitre.vn/thu-thuat-kien-thuc.htm',
                  'http://nhipsongso.tuoitre.vn/thi-truong.htm']
    travel_links = ['http://dulich.tuoitre.vn/goc-anh-lu-hanh.htm', 'http://dulich.tuoitre.vn/trai-nghiem-kham-pha.htm',
                    'http://dulich.tuoitre.vn/nhung-mien-dat-la.htm', 'http://dulich.tuoitre.vn/van-hoa.htm']

    print ('Initialize driver')
    domain = 'http://tuoitre.vn'
    display = Xvfb()
    display.start()
    driver = webdriver.Firefox()
    try:
        print ('Get topics links')
        topics_urls = get_topic_links(domain, topics)
        print ('Get articles links')
        get_article_fixed(['http://tuoitre.vn/tin/giao-duc'], 'Giáo dục', 'btn-readmore', 'div', 'highlight highlight-2', 20, False)
        driver.quit()
    except WebDriverException as we:
        print (we.msg)
    finally:
        display.stop()
    print ('Done')

