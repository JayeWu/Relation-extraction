import os
import re

dir_ = r'D:\rde\data\wangyi\网易证券\公司公告_文本'

keys = ['合营', '共同', '合作', '协力', '协同', '协作', '联合', '中外合资', '合资', '合办', '联营', '合股', '合伙', '联手', '共享', '共用', '合用', '牵手']
keys2 = ['组合', '并购', '并轨', '吞并', '归并', '合并']
keys3 = ['控股子公司', '下属', '子公司', '拥有', '旗下子公司', '下属子公司', '托管', '掌管', '控制', '所属', '附属', '隶属', '直属', '隶属于', '属于']
keys4 = ['出资', '设立', '重组为', '签订合资', '共同出资', '合作成立', '联合成立', '共建']
keys5 = ['持有股份', '持股', '认股', '持有', '股份', '股东', '投入', '投资', '股本', '转让股份', '控股', '持股人', '援款', '参股', '融资', '股权', '入股',
         '持仓', '回购']
keys6 = ['收买', '并购', '收购', '竞拍', '转让', '扩张', '购入', '购得', '购买', '支付', '承购', '并购', '注资', '整合', '买进', '买入', '赎买', '购销',
         '抛售', '售卖', '转售']
keys7 = ['持有']
keys8 = ['投资']
keys9 = ['转让']
keys10 = ['收购']
keys11 = ['共建']
count = 0

for path in os.listdir(dir_):
    path = os.path.join(dir_, path)
    keys = keys4
    for file in os.listdir(path):
        file = os.path.join(path, file)
        f = open(file, 'r', encoding='utf8')
        text = f.read()
        f.close()
        for key in keys:
            pattern = '[\u4e00-\u9fa5\n]*' + key + '[\u4e00-\u9fa5\n]*'
            count += len(re.findall(key, text))
            if re.search(pattern, text):
                s = re.findall(pattern, text)[0]
                print(file)
                print(s)
                print(key)
                print('===============================================')

print(count)