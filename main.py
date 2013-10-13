from matcher import matcher
import os.path
import imghdr
import requests
import subprocess

def getFrame():
    channel = 'therainman'
    r = requests.get('http://usher.justin.tv/find/%s.json?type=any' % (channel))
    result = r.json()[0]

    command = 'rtmpdump --swfUrl http://www.justin.tv/widgets/live_embed_player.swf --jtv \'%s\' --live -r %s/%s --stop 1 | avconv -i - -s 1920x1080 -vframes 1 file.png' % (result['token'], result['connect'], result['play'])

    subprocess.call(command, shell=True)


def matchChamp():
    getFrame()
    M = matcher()
    print(M.matchChamp('file.png'))

    # imdir = './img/test/'
    # imf = os.listdir(imdir)
    # M = matcher()
    # for ind,champFile in enumerate(imf):
    #     print(champFile)
    #     print(M.matchChamp(''.join([imdir,champFile])))



    # filePath = raw_input("Give me file name: ")
    # if(os.path.isfile(filePath) and imghdr.what(filePath)):
    #     M = matcher()
    #     print(M.matchChamp(filePath))
    # else:
    #     print("Please input a valid image file path")


if __name__ == "__main__":
    matchChamp()