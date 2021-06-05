import re

fp = open("store.txt",'r',encoding='utf-8')
processedFile = open("processed.txt",'w',encoding='utf-8')
emojiFile = open("emojiFile.txt",'w',encoding='utf-8')
for line in fp:
    line = " ".join(filter(lambda x:x[0]!='@', line.split()))
    line = " ".join(filter(lambda x:x[0:8]!="https://", line.split()))
    #line = " ".join(filter(lambda x:x, line.split(',')))
    if len(line)<2:
    	continue
    processedFile.write(line+"\n")
    print(line)
    emojis = re.findall(r'[^\w\⁠s,. ]', line)
    emojis = " ".join(emojis)
    '''lst =[]
    if (len(emojis)<2):
    	continue
    for emoji in emojis: 
    	if emoji in ['#','@','!',':','\'','\"','\\',",",'?','<','>','.',';','-','&','/']:
    		continue
    	lst.append(emoji)
    lst = " ".join(lst)
    emojiFile.write(lst+"\n")'''
    emojiFile.write(emojis+"\n")
fp.close()
processedFile.close()
emojiFile.close()