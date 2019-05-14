from tqdm import tqdm

alist = ['a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c',]
sum = 'c'
for i in tqdm(alist):
    sum += i
    print(sum)
print(sum)
