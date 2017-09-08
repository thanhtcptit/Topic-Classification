from tkinter import *
from tkinter import ttk
from get_data.crawf_text import crawf_from_url as crawf_url, crawf_from_disk as crawf_disk
from naive_bayes.data_preprocess import process_text

# Cửa số xử lý
class process_window():

    def __init__(self, type, content, clf):
        self.topics = ['Chính trị', 'Giáo dục', 'Công nghệ', 'Khác', 'Kinh tế', 'Pháp luật'
        , 'Thể thao', 'Văn hóa', 'Y tế']
        self.clean_text = ''
        self.result = 0
        self.result_pr = []
        self.processed_text = ''
        self.clf = clf
        self.content = content
        self.type = type
        self.top = Tk()
        self.top.title('Result')
        self.top.geometry('600x550')

        self.result_label = Label(self.top, text='Classify result: ')
        self.result_label.place(x=25, y=20)
        self.result_label.configure(font=("Times New Roman", 12))
        self.cat_label = Label(self.top, text='Topic: ')
        self.cat_label.place(x=25, y=50)
        self.cat_label.configure(font=("Times New Roman", 14))
        self.score_label = Label(self.top, text='Score: ')
        self.score_label.place(x=25, y=80)
        self.score_label.configure(font=("Times New Roman", 14))
        self.score_btn = Button(self.top, text='Show', command=self.score_btn_click)
        self.score_btn.place(x=80, y=80)

        self.notebook = ttk.Notebook(self.top)
        self.page1 = ttk.Frame(self.notebook)
        self.scroll1 = Scrollbar(self.page1)
        self.scroll1.pack(side=RIGHT, fill=Y)
        self.raw_text = Text(self.page1, width=75, yscrollcommand=self.scroll1.set)
        self.raw_text.pack(fill=BOTH)
        self.scroll1.configure(command=self.raw_text.yview)

        self.page2 = ttk.Frame(self.notebook)
        self.scroll2 = Scrollbar(self.page2)
        self.scroll2.pack(side=RIGHT, fill=Y)
        self.processed_text = Text(self.page2, width=75, yscrollcommand=self.scroll2.set)
        self.processed_text.pack()
        self.scroll2.configure(command=self.processed_text.yview)

        self.notebook.add(self.page1, text='Clean text')
        self.notebook.add(self.page2, text='Processed')
        self.notebook.place(x=25, y=125)

        self.confirm_btn = Button(self.top, text='Finish', command=self.finish_btn_click)
        self.confirm_btn.place(x=500, y=500)
        self.process()
        self.top.mainloop()

    # Xử lý đầu vào
    def process(self):
        if self.type == 0:
            text = crawf_url(self.content)
        elif self.type == 1:
            text = crawf_disk(self.content)
        else:
            text = self.content

        self.raw_text.insert('1.0', text)
        self.raw_text.configure(state='disabled')
        # Đánh dấu từ
        tag_text = process_text(text)
        self.processed_text.insert('1.0', tag_text)
        self.processed_text.configure(state='disabled')
        # Dự đoán dữ liệu thuộc topic
        self.result, self.result_pr = self.clf.classify(tag_text)
        self.cat_label.config(text='Topic: ' + self.topics[self.result])

    # Xem biểu đồ điểm xác suất
    def score_btn_click(self):
        self.clf.show_chart(self.result_pr, 'Topic', 'Score', 'Result', offset=50)

    # Kết thúc
    def finish_btn_click(self):
        self.top.destroy()



