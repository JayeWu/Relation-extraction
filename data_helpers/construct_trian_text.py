import os
import re

ceo_dir = r'D:\rde\data\search_by_baidu\co2ceo_sentences'
name_list_dir = r'D:\rde\data\search_by_baidu\names_list'
labeled_dir = r'../positive_labeled'
city_dir = r'D:\rde\data\search_by_baidu\city_sentences2'
creator_dir = r''
sentences_len = 50
neg_labeled_dir = r'D:\rde\enterprise_relation_extraction\negative_labeled'
famous_per_dir = r'D:\rde\enterprise_relation_extraction\out_data\famous_persones.txt'
big_cities_dir = r'D:\rde\enterprise_relation_extraction\out_data\big_cities.txt'


def cons_positive_trian_data_ceo():
    for path in os.listdir(ceo_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        ceo = ns[1]
        print(gs, ceo)
        file = os.path.join(ceo_dir, path)
        f = open(file, 'r', encoding='utf8')
        lines = f.readlines()
        f.close()
        f2 = open(os.path.join(labeled_dir, 'labeled_ceo2.txt'), 'a', encoding='utf8')
        for line in lines:
            new_txt = line.replace(gs, '<e1>' + gs + '</e1>').replace(ceo, '<e2>' + ceo + '</e2>')
            # print(new_txt)
            f2.write(new_txt)
        f2.close()


def cons_positive_trian_data_city():
    for path in os.listdir(city_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        city = ns[1]
        print(gs, city)
        file = os.path.join(city_dir, path)
        f = open(file, 'r', encoding='utf8')
        lines = f.readlines()
        f.close()
        f2 = open(os.path.join(labeled_dir, 'labeled_city.txt'), 'a', encoding='utf8')
        for line in lines:
            new_txt = line.replace(gs, '<e1>' + gs + '</e1>').replace(city, '<e2>' + city + '</e2>')
            # print(new_txt)
            f2.write(new_txt)
        f2.close()


def add_ceo_list():
    f = open(r'D:\rde\enterprise_relation_extraction\out_data\famous_persones.txt', 'a', encoding='utf8')
    for path in os.listdir(ceo_dir):
        ns = path.split('.txt')[0].split('+')
        ceo = ns[1]
        f.write(ceo.strip() + '\n')
    f.close()


def cons_negative_train_data_ceo():
    for path in os.listdir(ceo_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        ceo = ns[1]
        name_list_f_name = os.path.join(name_list_dir, gs + '.txt')
        f = open(os.path.join(ceo_dir, path), 'r', encoding='utf8')
        sentences = f.readlines()
        f.close()
        # if os.path.exists(name_list_f_name):
        f_n = open(famous_per_dir, 'r', encoding='utf8')
        persones = [x.strip() for x in f_n.readlines()]
        f_n.close()
        n_f = open(neg_labeled_dir + '/neg_labeled_ceo.txt', 'a', encoding='utf8')
        print(persones)
        for p in persones:
            if p == ceo:
                continue
            for sent in sentences:
                if p in sent:
                    sent = sent.strip()
                    sent = sent.replace(p, '<e2>' + p + '</e2>').replace(gs, '<e1>' + gs + '</e1>')
                    n_f.write(sent + '\n')
        n_f.close()


def cons_negative_train_data_city():
    city_dir = r'D:\rde\data\search_by_baidu\it_gs_baidu_search'
    f_kb = open(r'D:\rde\enterprise_relation_extraction\knowledge_base\property\ent_ceo.txt', 'r', encoding='utf8')
    lines = f_kb.readlines()
    f_kb.close()
    kb_set = dict()
    for line in lines:
        s = line.split('+')
        if s and len(s) >= 2:
            kb_set[s[0]] = s[1]
    for path in os.listdir(city_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        if gs not in kb_set:
            continue
        print(gs)
        city = kb_set[gs]
        print(city)
        # name_list_f_name = os.path.join(name_list_dir, gs + '.txt')
        n_f = open(neg_labeled_dir + '/ceo1.txt', 'a', encoding='utf8')
        ppp = os.path.join(city_dir, path)
        for file in os.listdir(ppp):
            file = os.path.join(ppp, file)
            if not os.path.isfile(file):
                continue
            f = open(file, 'r', encoding='utf8')
            sentences = f.readlines()
            f.close()
            # if os.path.exists(name_list_f_name):
            # f_n = open(big_cities_dir, 'r', encoding='utf8')
            f_n = open(r'D:\rde\enterprise_relation_extraction\out_data\famous_persones.txt', 'r', encoding='utf8')

            cities = [x.strip() for x in f_n.readlines()]
            f_n.close()
            for sent in sentences:
                sent = sent.strip()

                if gs not in sent:
                    continue
                for p in cities:
                    if p.strip() == city.strip():
                        continue
                    if p in sent:
                        sent1 = sent.replace(p, '<e2>' + p + '</e2>').replace(gs, '<e1>' + gs + '</e1>')
                        n_f.write(sent1 + '\n')
        n_f.close()


def cons_positive_train_data(data_type):
    _dir = r'D:\rde\data\search_by_baidu\year_news'
    for path in os.listdir(_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        tp = ns[1]
        print(gs, tp)
        _dir2 = os.path.join(_dir, path)
        for file in os.listdir(_dir2):
            file = os.path.join(_dir2, file)
            if not os.path.isfile(file):
                continue
            f = open(file, 'r', encoding='utf8')
            lines = f.readlines()
            f.close()
            f2 = open(os.path.join(labeled_dir, data_type + '.txt'), 'a', encoding='utf8')
            gs_short = gs.replace('股份有限公司', '').replace('责任有限公司', '')
            gs_short = re.sub('[\u4e00-\u9fa5]+市', '', gs_short)
            print(gs_short)
            for line in lines:
                if gs_short in line and tp in line:
                    new_txt = line.replace(gs, '<e1>' + "&&&&&" + '</e1>').replace(tp, '<e2>' + tp + '</e2>') \
                        .replace(gs_short, '<e1>' + "&&&&&" + '</e1>').replace("&&&&&", gs)
                    f2.write(new_txt)
            f2.close()


def cons_negative_train_data(data_type):
    _dir = r'D:\rde\data\search_by_baidu\year_news'
    for path in os.listdir(_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        tp = ns[1]
        print(gs, tp)
        _dir2 = os.path.join(_dir, path)
        for file in os.listdir(_dir2):
            file = os.path.join(_dir2, file)
            if not os.path.isfile(file):
                continue
            f = open(file, 'r', encoding='utf8')
            lines = f.readlines()
            f.close()
            f2 = open(os.path.join(neg_labeled_dir, data_type + '.txt'), 'a', encoding='utf8')
            gs_short = gs.replace('股份有限公司', '').replace('责任有限公司', '')
            print(gs_short)
            for line in lines:
                gs_short = re.sub('[\u4e00-\u9fa5]+市', '', gs_short)
                _tps = re.findall('\d{4}年', line)
                if _tps and gs_short in line:
                    for tp1 in _tps:
                        if tp1 == tp:
                            continue
                        new_txt = line.replace(gs, '<e1>' + "&&&&&" + '</e1>').replace(tp1, '<e2>' + tp1 + '</e2>') \
                            .replace(gs_short, '<e1>' + "&&&&&" + '</e1>').replace("&&&&&", gs)
                        f2.write(new_txt)
            f2.close()


def delete_disquality_data(file, file2):
    f = open(file, 'r', encoding='utf8')
    f2 = open(file2, 'a', encoding='utf8')
    for line in f.readlines():
        if 'e1' in line and 'e2' in line:
            f2.write(line)


def cons_positive_train_data_year_from_search_news(data_type):
    _dir = r'D:\rde\data\search_by_baidu\news'
    f_kb = open(r'D:\rde\enterprise_relation_extraction\knowledge_base\property\ent_year.txt', 'r', encoding='utf8')
    lines = f_kb.readlines()
    f_kb.close()
    kb_set = dict()
    for line in lines:
        s = line.split('+')
        if s and len(s) >= 2:
            kb_set[s[0]] = s[1]

    for path in os.listdir(_dir):
        ns = path.split('.txt')[0].split('+')
        gs = ns[0]
        if gs not in kb_set:
            continue
        tp = kb_set[gs]
        print(gs, tp)
        _dir2 = os.path.join(_dir, path)
        for file in os.listdir(_dir2):
            file = os.path.join(_dir2, file)
            if not os.path.isfile(file):
                continue
            f = open(file, 'r', encoding='utf8')
            lines = f.readlines()
            f.close()
            f2 = open(os.path.join(labeled_dir, data_type + '1.txt'), 'a', encoding='utf8')

            gs_short = gs.replace('股份有限公司', '').replace('责任有限公司', '')
            for line in lines:
                gs_short = re.sub('[\u4e00-\u9fa5]+市', '', gs_short)
                if gs_short in line and tp in line:
                    new_txt = line.replace(gs, '<e1>' + "&&&&&" + '</e1>').replace(tp, '<e2>' + tp + '</e2>') \
                        .replace(gs_short, '<e1>' + "&&&&&" + '</e1>').replace("&&&&&", gs)
                    f2.write(new_txt)
            f2.close()


def get_rela_by_regular(keys, rela, dirs):
    for dir_ in os.listdir(dirs):
        # gs = str(dir_.split('+')[0])
        # gs = dir_
        gs = str(dir_.split('(')[0])
        gs_short = gs.replace('股份有限公司', '').replace('责任有限公司', '')
        gs_short = re.sub('[\u4e00-\u9fa5]+市', '', gs_short)
        dir_ = os.path.join(dirs, dir_)
        contents = []
        for file in os.listdir(dir_):
            if not os.path.isfile(os.path.join(dir_, file)):
                continue
            f = open(os.path.join(dir_, file), 'r', encoding='utf8')
            for line in f.readlines():
                if gs_short in line:
                    for key in keys:
                        if key in line:
                            idx = line.index(key)
                            scope1 = idx - 100 if idx > 100 else 0
                            scope2 = idx + 100 if idx < len(line) - 100 else len(line) - 1
                            ct = line[scope1:scope2]
                            print(ct)
                            contents.append(ct)
                            break
        if not contents:
            continue
        f2 = open(os.path.join(r'D:\rde\data\relas_txt1', gs + '.txt'), 'a', encoding='utf8')
        for content in contents:
            f2.write(content+'\n')
        f2.close()


def remove_oth():
    file = r'D:\rde\enterprise_relation_extraction\negative_labeled\city.txt'
    file2 = r'D:\rde\enterprise_relation_extraction\negative_labeled\city2.txt'
    f = open(file,'r',encoding='utf8')
    f2 = open(file2,'w',encoding='utf8')
    lines = f.readlines()
    for i in range(len(lines)):
        if i <9122:
            f2.write(lines[i])
    f.close()
    f2.close()


if __name__ == '__main__':
    # keys = ['合营', '共同', '合作', '协力', '协同', '协作', '联合', '中外合资', '合资', '合办', '联营', '合股', '合伙', '联手', '共享', '共用', '合用',
    #         '牵手']
    #
    # get_rela_by_regular(keys, '合作', r'D:\rde\data\wangyi\网易证券\公司公告_文本')
    # js(r'D:\rde\enterprise_relation_extraction\negative_labeled\neg_labeled_city.txt',
    #    r'D:\rde\enterprise_relation_extraction\negative_labeled\city.txt')
    # remove_oth()

    cons_negative_train_data_city()
