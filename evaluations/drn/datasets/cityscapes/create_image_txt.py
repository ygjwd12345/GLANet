import os
path='/data2/gyang/TAGAN/results/cityscape/best/'
# f = open("val_images.txt", "w")
list=[]
for i in range(500):
    print(i)
    name=str(i)+'_A.png'
    list.append(name)
    print(name)
    # path=os.path.join(path,name)
    # print(path)
print(list[0])
print(len(list))
with open('val_images.txt', 'w') as f:
    for item in list:
        f.write("%s%s\n" % (path,item))