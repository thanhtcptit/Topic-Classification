import requests, re, os
from bs4 import BeautifulSoup as bs
from utils import *

# Tìm xâu con trong xâu
def find_all_ss(patern, string):
    return [m.start() for m in re.finditer(patern, string)]

# Tìm vị trí ký tự kết thúc
def end_index(from_idx, string):
    end_idx = '>'
    for i in range(from_idx + 1, len(string)):
        if string[i] == end_idx:
            return i

# Lọc các tag không cần thiết
def filter_tag(raw_text):
    text = []
    break_flag = False
    # Các xâu kết thúc file
    stop_pattern = ['Vui lòng nhập Email', 'Thông tin bạn đọc', 'Vui lòng nhập nội dung bình luận']
    for t in raw_text:
        open_char = 0
        line = str(t)
        line = line.replace('&gt;', '')
        processed_line = ''
        for sp in stop_pattern:
            if line.find(sp) > 0:
                break_flag = True
                break
        if break_flag:
            break
        for i in range(0, len(line)):
            # Đánh dấu 1 tag
            if line [i] == '<' and line [i+1] != ' ':
                open_char = open_char + 1
                continue
            # Kết thúc 1 tag
            elif line [i] == '>' and line [i-1] != ' ':
                open_char = open_char - 1
                continue
            # Đọc ký tự
            elif open_char == 0:
                processed_line += line[i]
        if len(processed_line) > 0:
            text.append(processed_line)
    return '\n'.join(text)

# Tải về file html
def html_download():
    session = requests.session()
    adapter = requests.adapters.HTTPAdapter(max_retries=10)
    session.mount('http://', adapter)

    for topic in topics:
        file = HTML_LINK_DIR + topic
        html_dir = HTML_DIR + topic + '/'
        if os.path.exists(html_dir) == False:
            os.mkdir(html_dir)
        with open(file, 'r') as f:
            urls = f.read().strip().split('\n')
            for i, url in enumerate(urls):
                r = session.get(url.strip())
                html = bs(r.content, 'lxml')
                html_file = html_dir + str(i + 1)
                if os.path.exists(html_file):
                    continue
                with open(html_file, 'w') as hf:
                    hf.write(str(html))

# Lấy dữ liệu từ 1 topic lưu trong máy
def crawf_text_from_topic(topic):
    topic_dir = HTML_DIR + topic + '/'
    for file_name in os.listdir(topic_dir):
        file_path = topic_dir + file_name
        with open(file_path, 'r') as f:
            html = bs(f.read(), 'lxml')
            raw = (html.find_all('title')) + html.find_all('p')
            text = filter_tag(raw)
            text[0] = text[0][:-18] + '\n'
            del text[1]
            if len(text) <= 1:
                print('ignore')
                continue
            p = ARTICLE_DIR + topic + '/' + str(file_name)
            file = open(p, 'w')
            file.write(text[0] + '\n')
            for i in range(1, len(text)):
                if len(text[i]) < 3:
                    continue
                file.write(text[i].strip() + '\n')
            file.close()

# Đọc dữ liệu từ 1 trang web
def crawf_from_url(url):
    session = requests.session()
    adapter = requests.adapters.HTTPAdapter(max_retries=10)
    session.mount('http://', adapter)
    r = session.get(url.strip())
    html = bs(r.content, 'lxml')
    raw = (html.find_all('title')) + html.find_all('p')
    text = filter_tag(raw)
    return text

# Đọc dữ liệu trong máy
def crawf_from_disk(path):
    with open(path, 'r') as f:
        return f.read().strip()

#######################
if __name__ == '__main__':
    topics =  ['Chính trị', 'Giáo dục', 'Kinh tế', 'Pháp luật', 'Văn hóa', 'Y tế', 'Khác', 'Khoa học công nghệ', 'Thể thao']

