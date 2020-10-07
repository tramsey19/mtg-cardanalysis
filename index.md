# Magic: The Gathering Card Analysis

Magic: The Gathering (Magic) is an incredibly complex card game.  Commander is one of the most popular formats in Magic.  One of the issues Commander currently faces is identifying the power level of your deck.  Commander is typically played with 4 players, each piloting their own deck.  If one player has an incredibly powerful deck while the others are playing more casual decks that player has a huge advantage.  Identifying the power level of your deck isn't as easy as just looking at the cards.  Here, I've created a model to predict the power level of Magic cards and subsequent Commander decks.

The first step is to create a csv file of a few commanders you'd like to analyze.  I've uploaded an example [here](https://github.com/tramsey19/mtg-cardanalysis/blob/master/commanders.csv) but feel free to add/remove any you'd like to see.

### Data Collection
```python
import os 
import requests
import pandas as pd
from pandas import json_normalize
from bs4 import BeautifulSoup as bs
import time

# Gather GUIDs for decks from commanders.csv
df = pd.read_csv('commanders.csv',header=None)

url = 'https://edhrec-json.s3.amazonaws.com/en/listofdecks/'
for index,rows in df.iterrows():
    path = 'Decklists\\' + rows[0] +'\\'
    os.mkdir(path)
    
    req = url + rows[0] + '.json'
    water = requests.get(req)
    soup = bs(water.content)
    
    with open(path+rows[0]+'.csv','wb') as file:
        for a in soup.find_all('a', href=True):
            file.write(a['href'].replace('\\"/deckpreview/','').replace('\\"','\r\n').encode('utf-8'))

# Base URLs for scraping
edhrec = 'https://edhrec.com/api/deckpreview/'
scryfall = 'https://api.scryfall.com/cards/collection'

# Using the GUIDs gathered from above, this section scrapes the decklists from EDHREC, 
# then passes the lists to Scryfall to gather the individual card data.
for subdir,dirs,files in os.walk('Decklists\\'):
    commander = subdir.replace('Decklists\\','')
    savepath = subdir + '\\'
    for file in files:
        df = pd.read_csv(os.path.join(subdir,file),header=None)
        for index,rows in df.iterrows():
            url = edhrec + rows[0]
            arr = []
            arr2 = []
            headers = {'Content-Type':'application/json'}
            # We are only targeting specific card attributes
            cols = ['name', 'mana_cost', 'cmc', 'type_line', 'oracle_text', 'colors', 
                   'color_identity', 'keywords','rarity', 'edhrec_rank', 'prices.usd', 
                   'prices.usd_foil','produced_mana','loyalty', 'power', 'toughness']

            res = requests.get(url)

            for a in res.json()['cards']:
                obj = {'name':a}
                if len(arr) < 50:
                    arr.append(obj)
                else:
                    arr2.append(obj)

            ident1 = {'identifiers':arr}
            ident2 = {'identifiers':arr2}

            res2 = requests.post(scryfall,json=ident1,headers=headers)
            dFirst = json_normalize(res2.json()['data'])
            time.sleep(.1)
            if (len(arr2)>0):
                res3 = requests.post(scryfall,json=ident2,headers=headers)
                dSec = json_normalize(res3.json()['data'])
                df = pd.concat([dFirst,dSec],ignore_index=True)
            else:
                df = dFirst

            for b in cols:
                if b in dFirst.columns:
                    continue
                else:
                    df[b] = ''
                    
            df = df[cols]
            savename = commander +'-'+ rows[0]
            df.to_csv(savepath+savename+'.csv')
```
