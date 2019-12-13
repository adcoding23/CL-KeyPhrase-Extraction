#import glob   
mypath = '/home/shreya/Documents/CMSC723/yutaya/EmbedRank/embedrank/Hulth2003/Training'
write_path =  '/home/shreya/Documents/CMSC723/yutaya/EmbedRank/embedrank/Hulth2003/Tokenized_Training/'#give the path that has all the files   
#files=glob.glob(path)   
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:   
    if(file.endswith(".abstr")):
        f = open(mypath+"/"+file,'r')
        f_doc = f.read()
        #f_new = f_doc.split()
        #f_doc = "['"+str(f_doc)+"']"
        f_doc = str(f_doc)+'$'
        #f.close()
        #f = open(file+"_tok.txt", "w")#make another directory to store the tokenized files. Not overwritting the original text files
        #f.write(str(f_new))
        f.close()
        f = open(write_path+file+"_tok", "w")#make another directory to store the tokenized files. Not overwritting the original text files
        f.write(f_doc)
        f.close()
    