import java.io.File;
import java.io.IOException;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Example program to list links from a URL.
 */
public class Parse {
    public static void main(String[] args) throws IOException {
    	File in = new File("index.html");
    	//<img data-original="http://ddragon.leagueoflegends.com/cdn/3.12.34/img/champion/Akali.png" style="display: inline;" class="img" src="http://ddragon.leagueoflegends.com/cdn/3.12.34/img/champion/Akali.png">
    	Document doc = Jsoup.parse(in, null);
    	//Elements champions = doc.select("img[src*='http://ddragon.leagueoflegends.com']");
        //Elements champions = doc.select("[id*=\"champion-grid\"] > div > span > a > img");
    	Elements champions = doc.select("[id*=\"champion-grid\"]");
        Elements names = doc.select(".champ-name > a");

        if(champions.isEmpty()){
        	System.out.println("Champions is empty");
        }
        if(names.isEmpty()){
        	System.out.println("names is empty");
        }
        
        for (Element champion : champions) {
        	System.out.println(champion.getAllElements().toString());
        }
        
        for (Element name : names) {
            System.out.println(name.getAllElements().toString());
        }

    }
}
