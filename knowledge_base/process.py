import os
import re


def fun_ent_ceo_city():
    files = os.listdir(r'D:\rde\data\search_by_baidu\city_sentences2')
    f = open(r'./property/ent_city.txt', 'a', encoding='utf8')
    for file in files:
        line = file.replace('.txt', '')
        f.write(line + '\n')


def fun_ent_founder():
    dir_ = r'D:\rde\data\text\baidu_brief_text'
    files = os.listdir(dir_)
    pattern = '(?<=创[始办]人:)[\u4e00-\u9fa5]*'
    f = open(r'./property/ent_founder.txt', 'a', encoding='utf8')
    for file in files:
        f_r = open(os.path.join(dir_, file), 'r', encoding='utf8')
        txt = f_r.read()
        f_r.close()
        ent = file.split('_百')[0]

        founder = re.findall(pattern, txt)
        if founder:
            f.write(ent + '+' + founder[0] + '\n')
    f.close()


def fun_ent_year():
    dir_ = r'D:\rde\data\text\baidu_brief_text'
    pattern = '(?<=成立时间:)\d{4}'
    files = os.listdir(dir_)
    f = open(r'./property/ent_year.txt', 'a', encoding='utf8')
    for file in files:
        f_r = open(os.path.join(dir_, file), 'r', encoding='utf8')
        txt = f_r.read()
        f_r.close()
        ent = file.split('_百')[0]
        founder = re.findall(pattern, txt)
        if founder:
            f.write(ent + '+' + founder[0] + '年\n')
    f.close()


if __name__ == '__main__':
    fun_ent_year()
