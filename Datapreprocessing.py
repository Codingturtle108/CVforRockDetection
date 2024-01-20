import os
path = 'opencv/Dataset'
for i in os.listdir(path):
    subpath = os.path.join(path,i)
    count  =1
    for j in os.listdir(subpath):
        src_file = os.path.join(subpath,j)
        new_filename = f'{i}train{count}.jpg'
        dst_file =  os.path.join(subpath, new_filename)
        os.rename(src_file,dst_file)
        count +=1