## Magic: The Gathering Card Analysis

View the [README](https://github.com/tramsey19/mtg-cardanalysis/blob/master/README.md) to get started.

Or if you're feeling adventurous, jump right in to the [EDHREC Data Collection](https://github.com/tramsey19/mtg-cardanalysis/blob/master/EDHREC%20Data%20Collection.ipynb).
First, scrape your unique deck IDs for each commander listed in the csv file.  
```python
df = pd.read_csv('commanders.csv',header=None)
url = 'https://edhrec-json.s3.amazonaws.com/en/listofdecks/'        
```
The for-loop will iterate through each commander, scrape the unique ID from the json file, and write the GUID to a csv
```python
file.write(a['href'].replace('\\"/deckpreview/','').replace('\\"','\r\n').encode('utf-8'))
```
The next section will go through each GUID, make a request to EDHREC to get the card list, then a subsequent call to Scryfall to get the data for each card.

If you've already gathered the data, you can check out the [deep learning model](https://github.com/tramsey19/mtg-cardanalysis/blob/master/Card%20Analysis.ipynb) that will analyze the oracle text on all cards and make a prediction of powerful/not powerful (with 99.5% accuracy!). 
```python
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001), input_shape = (5402,)))
model.add(layers.Dropout(0.2))
model.add(BatchNormalization())
model.add(layers.Dense(16, kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 = 0.001), activation = 'relu'))
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
```

Or if you'd rather look at a specific commander, you can analyze the power level of a specfic deck and compare it to the average power level of that commander with the [Commander Analysis](https://github.com/tramsey19/mtg-cardanalysis/blob/master/Commander%20Analysis.ipynb) file.
```python
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
```
