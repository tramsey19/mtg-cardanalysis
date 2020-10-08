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
typerank = {'Enchantment':9,'Artifact':8,'Planeswalker':7,'Creature':5,
    'Instant':7,'Sorcery':1,'Land':10}

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
                        if dfrows['oracle_text'].lower().\
                            find(abilrows['ability'].lower()) > -1:
                            ability_sum += abilrows['rank']
                    except:
                        continue
                deck.at[dfindex,'ability_score'] = ability_sum

            dfclean = deck[['cmc','type_line','loyalty','power',
                'toughness','ability_score']].copy()
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
            dfclean['ability_score'] = \
                dfclean['ability_score']/dfclean.apply(max)['ability_score']
            dfclean['type_line'] = \
                dfclean['type_line']/dfclean.apply(max)['type_line']
            dfclean['loyalty'] = \
                dfclean['loyalty']/dfclean.apply(max)['loyalty']
            dfclean['power'] = dfclean['power']/dfclean.apply(max)['power']
            dfclean['toughness'] = \
                dfclean['toughness']/dfclean.apply(max)['toughness']
            dfclean['cmc'] = 1/(dfclean['cmc']+1)
            dfclean=dfclean.fillna(0)

            dfclean['card_power']=dfclean.sum(axis=1)/dfclean.\
                astype(bool).sum(axis=1)

            powerList.append(dfclean['card_power'].sum())


atraxa = []
commanderPower(atraxa,'Decklists/atraxa-praetors-voice/',
    'atraxa-praetors-voice.csv')

breya = []
commanderPower(breya, 'Decklists/breya-etherium-shaper/',
    'breya-etherium-shaper.csv')

sram = [] 
commanderPower(sram, 'Decklists/sram-senior-edificer/',
    'sram-senior-edificer.csv')

zur = [] 
commanderPower(sram, 'Decklists/zur-the-enchanter/',
    'zur-the-enchanter.csv')
```
The last few lines create lists containing deck powers.  With these we can create some bar charts to see the data a little better and see how these commanders rank up against each other.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/commanders.png)

This graph shows that Sram appears to be the weakest while Zur and Breya appear stronger, but it is difficult to tell.  The below graphs break out the decks into individual graphs.  These graphs are separated into 5 groups with static ranges from the min to max for each specific commander.  Pay close attention to the X-axis.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/atraxa.png)

Atraxa, for example, has the majority of decks with a power level between 48-55.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/breya.png)

Breya is skewed a little to the right, with most of her decks ranging from 50-56.  It appears Breya might be a little stronger than Atraxa.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/sram.png)

Sram is skewed left with most decks ranging from 35-41.  Sram appears to be the least powerful commander.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/zur.png)

Zur has an interesting spread, most decks are between 45-57.  It looks like a bit wider spread might help clear things up.

![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/zur10.png)

With 10 groups instead of 5 we can see most the Zur decks range from 57-64, which would make this the most powerful commander.

In general, this shows that we are directionally correct with our power rankings.  Zur is considered a top-tier commander with Breya following close behind.  Sram is a mono-white commander making it one of the least powerful commanders to build around.  Atraxa is used across a wide variety of decks, but none of which are as powerful as Breya or Zur.  It makes sense for her rankings to fall in the middle of the line.

In addition to looking at specific commanders/cards, we can also look at the cards in general and create a deep learning model to analyze the power level of cards.  First, we need to gather the card data which can be found under the "Oracle Cards" section here:  https://scryfall.com/docs/api/bulk-data.

### Card Analysis
```python
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.callbacks import EarlyStopping
from keras import backend
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding

# Rename the "oracle-cards" file to "rulings.json"
df = pd.read_json('rulings.json')
dfabilities = pd.read_csv('abilities',header=None,names=['ability','rank'])

# Create ability score column
df['ability_score'] = 0
for dfindex,dfrows in df.iterrows():
    ability_sum = 0
    for abilindex,abilrows in dfabilities.iterrows():
        try:
            if dfrows['oracle_text'].find(abilrows['ability']) > -1:
                ability_sum += abilrows['rank']
        except:
            continue
    df.at[dfindex,'ability_score'] = ability_sum
 
# Create an ability index
ability_index = {}
count = 0
for index,rows in df.iterrows():
    try:
        words = rows['oracle_text'].lower().replace(rows['name'].lower(),'this object').split()
        for a in words:
            a = ''.join(b for b in a if b not in ['.','(',')'])
            if a not in ability_index:
                count += 1
                ability_index[a] = count
            else:
                continue
    except:
        continue
        
# Create train/test data
train_data = []
train_labels = []
test_data = []
test_labels = []
df = df.fillna(0)
for index,rows in df.iterrows():
    try:
        words = rows['oracle_text'].lower().replace(rows['name'].lower(),'this object').split()
        indices = []
        for a in words:
            a = ''.join(b for b in a if b not in ['.','(',')'])
            indices.append(ability_index[a])
        # Split train/test into 80/20
        if len(train_data) <= len(df)*.8:
            train_data.append(indices)
            train_labels.append(rows['ability_score'])
        else:
            test_data.append(indices)
            test_labels.append(rows['ability_score'])
    except:
        continue        

# Convert data to NumPy arrays
np_train_data = np.array(train_data)
np_train_labels = np.array(train_labels)
np_test_data = np.array(test_data)
np_test_labels = np.array(test_labels)

# Convert labels to floats
np_train_labels = np_train_labels.astype(np.float)
np_test_labels = np_test_labels.astype(np.float)

# Normalize the labels between 0 and 1
np_train_labels = np_train_labels/np_train_labels.max()
np_test_labels = np_test_labels/np_test_labels.max()

# Set each label to 0 or 1
count = 0
for a in np_train_labels:
    # Adjust this value higher/lower to be more/less strict on what is considered good/bad
    if a < .5:
        np_train_labels[count] = 0
    else:
        np_train_labels[count] = 1
    count += 1
count = 0
for a in np_test_labels:
    # Adjust this value higher/lower to be more/less strict on what is considered good/bad
    if a < .5:
        np_test_labels[count] = 0
    else:
        np_test_labels[count] = 1
    count += 1
    
# This function will create a tensor that is 17537 by 5402. 17537 is the number of samples
# and 5402 is the number of unique words. The tensor will have all zeros except for ones where that word is in the ability
def vectorize_sequences(sequences, dimension=5402):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
    
# Apply the vectorize function to the train_data and test_data
x_train = vectorize_sequences(np_train_data)
x_test = vectorize_sequences(np_test_data)


# View the train shape, values, and dimensions
print(x_train.shape)
print(x_train[0])
print(x_train.ndim)

y_train = np.asarray(np_train_labels).astype('float32')
y_test = np.asarray(np_test_labels).astype('float32')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

np.random.seed(42)
backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (5402,)))
model.add(layers.Dropout(0.2))
model.add(BatchNormalization())
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(BatchNormalization())
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x_train,
                   y_train,
                   epochs = 20,
                   batch_size = 500,
                   validation_data = (x_val, y_val),
                   callbacks=[EarlyStopping(monitor='val_acc', patience=3, restore_best_weights = True)])

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(results)
```
![](https://raw.githubusercontent.com/tramsey19/mtg-cardanalysis/master/assets/deepmodel.png)

This model achieved 99.52% test accuracy.  This means the model can estimate with near perfect accuracy if a card is considered "powerful" or not.  

Here is another method I found and adapted from this website:
#### https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
```python
docs = df['oracle_text'].copy()

# Convert all non-str values into empty strings
count = 0
for a in docs:
    try:
        a.lower()
    except:
        docs[count] = ''
    finally:
        count += 1
        
# Define labels
labels = df['ability_score'].copy()

# Normalize the labels between 0 and 1
labels = labels/labels.max()

# Set each label to 0 or 1
count = 0
for a in labels:
    # Adjust this value higher/lower to be more/less strict on what is considered good/bad
    if a < .5:
        labels[count] = 0
    else:
        labels[count] = 1
    count += 1      
    
# Vocab size should be greater than number of unique words to prevent collisions
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# Max Length is maximum number of words allowed in an ability
max_length = 20
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define the model
model = Sequential()
# 16 is the dimensions (features/attributes) for a specific document
model.add(Embedding(vocab_size, 16, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))    
```        
This model achieved 99.98% accuracy when I ran it.  While I prefer the first model over the second, its a bit like using a sledgehammer on a nail.  The second model is much more efficient and provides a greater accuracy, so I would probably use it going forward.
