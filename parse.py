import urllib2
import bs4

def getImgLink(webpage):
    request = urllib2.Request(webpage)
    page = urllib2.urlopen(request)
    soup = bs4.BeautifulSoup(page)
    tags = soup.findAll('img')
    print "\n".join(set(tag['src'] for tag in tags))

if __name__ == "__main__":
    getImgLink("http://gameinfo.na.leagueoflegends.com/en/game-info/champions/")