Magic: The Gathering (Magic) is an incredibly complex card game.  Commander is one of the most popular formats in Magic.  One of the issues Commander currently faces is identifying the power level of your deck.  Commander is typically played with 4 players, each piloting their own deck.  If one player has an incredibly powerful deck while the others are playing more casual decks that player has a huge advantage.  Identifying the power level of your deck isn't as easy as just looking at the cards.  Here, I've created a model to predict the power level of Magic cards and subsequent Commander decks.

The first step is to create a csv file of a few commanders you'd like to analyze.  I've uploaded an example [here](https://github.com/tramsey19/mtg-cardanalysis/blob/master/commanders.csv) but feel free to add/remove any you'd like to see.  Save the csv file as "commanders.csv" then run the below code in the same directory.

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
os.mkdir('Decklists\\')
for index,rows in df.iterrows():
    path = 'Decklists\\' + rows[0] +'\\'
    os.mkdir(path)
    
    req = url + rows[0] + '.json'
    water = requests.get(req)
    soup = bs(water.content)
    
    with open(path+rows[0]+'.csv','wb') as file:
        for a in soup.find_all('a', href=True):
            file.write(a['href'].replace('\\"/deckpreview/','')\
                .replace('\\"','\r\n').encode('utf-8'))

# Base URLs for scraping
edhrec = 'https://edhrec.com/api/deckpreview/'
scryfall = 'https://api.scryfall.com/cards/collection'

# Using the GUIDs gathered from above, this section scrapes the decklists 
# from EDHREC then passes the lists to Scryfall to gather the card data.
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
            cols = ['name', 'mana_cost', 'cmc', 'type_line', 'oracle_text', 
                   'colors','color_identity', 'keywords','rarity', 'edhrec_rank',
                   'prices.usd', 'prices.usd_foil','produced_mana','loyalty',
                   'power', 'toughness']

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

After collecting the data, we need to create an abilities file.  I've created one [here](https://github.com/tramsey19/mtg-cardanalysis/blob/master/abilities).  Feel free to modify to your liking.  This file is a comma separated file of oracle text as it would appear on a card and the corresponding power level for that text.  The power rankings assigned were assigned based off my own knowledge of the game and the meta I play in.  They are highly subjective.  With the abilities file and the commander decklists, we can now perform analysis on the commander decks.

### Commander Deck Power/Analysis
```python
import pandas as pd
import os

# Both the abilities file and the typerank are highly subjective.  
# They were created using a combination of ngrams as well as my own 
# personal knowledge of the game.  These should be modified as you see fit.
dfabilities = pd.read_csv('abilities',header=None,names=['ability','rank'])
typerank = {'Enchantment':9,'Artifact':8,'Planeswalker':7,'Creature':5,'Instant':7,'Sorcery':1,'Land':10}

# Create a function for reusability
def commanderPower(powerList, DecklistDir, skipFile):
    for subdir,dirs,files in os.walk(DecklistDir):
        for file in files:
            if skipFile in file:
                continue
            deck = pd.read_csv(os.path.join(subdir,file))
            for dfindex,dfrows in deck.iterrows():
                ability_sum = 0
                for abilindex,abilrows in dfabilities.iterrows():
                    try:
                        if dfrows['oracle_text'].lower().find(abilrows['ability'].lower()) > -1:
                            ability_sum += abilrows['rank']
                    except:
                        continue
                deck.at[dfindex,'ability_score'] = ability_sum

            dfclean = deck[['cmc','type_line','loyalty','power','toughness','ability_score']].copy()
            for index,rows in dfclean.iterrows():
                typeline = rows['type_line']
                typewords = typeline.split()
                flag = True
                for a in typewords:
                    if a in typerank.keys():
                        dfclean.at[index,'type_line'] = typerank[a]
                        flag = False
                if flag:
                    dfclean.at[index,'type_line'] = 0
            dfclean=dfclean.fillna(0)

            # Variables like * and X need to be converted to integers
            for index,rows in dfclean.iterrows():
                if '*' in str(rows['loyalty']) or rows['loyalty'] == 'X':
                    dfclean.at[index,'loyalty']=0
                if '*' in str(rows['power']) or rows['power'] == 'X':
                    dfclean.at[index,'power']=0
                if '*' in str(rows['toughness']) or rows['toughness'] == 'X':
                    dfclean.at[index,'toughness']=0

            # Columns need to be converted to ints
            dfclean['power'] = dfclean['power'].astype(int)
            dfclean['toughness'] = dfclean['toughness'].astype(int)
            dfclean['loyalty'] = dfclean['loyalty'].astype(int)

            # Normalize values by dividing by max
            dfclean['ability_score'] = dfclean['ability_score']/dfclean.apply(max)['ability_score']
            dfclean['type_line'] = dfclean['type_line']/dfclean.apply(max)['type_line']
            dfclean['loyalty'] = dfclean['loyalty']/dfclean.apply(max)['loyalty']
            dfclean['power'] = dfclean['power']/dfclean.apply(max)['power']
            dfclean['toughness'] = dfclean['toughness']/dfclean.apply(max)['toughness']
            dfclean['cmc'] = 1/(dfclean['cmc']+1)
            dfclean=dfclean.fillna(0)

            dfclean['card_power']=dfclean.sum(axis=1)/dfclean.astype(bool).sum(axis=1)

            powerList.append(dfclean['card_power'].sum())


atraxa = []
commanderPower(atraxa,'Decklists/atraxa-praetors-voice/','atraxa-praetors-voice.csv')
```
![](https://github.com/tramsey19/mtg-cardanalysis/assets/test.png)
```python
breya = []
commanderPower(breya, 'Decklists/breya-etherium-shaper/','breya-etherium-shaper.csv')

sram = [] 
commanderPower(sram, 'Decklists/sram-senior-edificer/','sram-senior-edificer.csv')

```
