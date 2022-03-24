import codecs

recordFile = '1.log'
FoundFlag = False
fileObj = codecs.open(recordFile, 'r+', 'utf-8')
lineTemp = fileObj.readlines()
count_1 = 1
count_2 = 1
server_label_1 = "### Time Windows ###"
server_label_2 = "# Start time"

for line in lineTemp:
    # 等于-1表示匹配不到，非-1表示匹配到的索引
    if line.strip().find(server_label_1) == -1:
        FoundFlag = False
        count_1 += 1
        # print("the line is: " + line, end='')
    else:
        break

for line in lineTemp:
    # 等于-1表示匹配不到，非-1表示匹配到的索引
    if line.strip().find(server_label_2) == -1:
        FoundFlag = False
        count_2 += 1
        # print("the line is: " + line, end='')
    else:
        break

fileObj.close()


