from matcher import matcher
import os.path
import imghdr

def matchChamp():
    filePath = raw_input("Give me file name: ")
    # imdir = './img/test/'
    # imf = os.listdir(imdir)
    # M = matcher()
    # for ind,champFile in enumerate(imf):
    #     print(champFile)
    #     print(M.matchChamp(''.join([imdir,champFile])))
    if(os.path.isfile(filePath) and imghdr.what(filePath)):
        M = matcher()
        print(M.matchChamp(filePath))
    else:
        print("Please input a valid image file path")


if __name__ == "__main__":
    matchChamp()