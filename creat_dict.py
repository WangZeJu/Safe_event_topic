# ./usr/bin/python
# coding = utf-8

keywords = ''
for i in range(1,2619):
    with open("/home/wangzeju/桌面/lunwen/keywords/" + str(i) + ".txt") as f:
         keywords = keywords + ';' + f.read()
         f.close()

keywords_list_ = keywords.split(';')
keywords_list = []
for keyword_ in keywords_list_:
    if keyword_ not in keywords_list and keyword_ != '':
       keywords_list.append(keyword_)

for keyword in keywords_list:
    with open("/home/wangzeju/桌面/lunwen/self_dict.txt","ab+") as save_f:
         save_f.write((keyword + '\n').encode())
      
