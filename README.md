# mtg-cardanalysis
Analyze oracle text of Magic: The Gathering cards and create a model to predict power level

DATA COLLECTION:
    This file reads in a list of commanders from the "commanders.csv" file.  The commander names need to be in the format expected by edhrec.  For example:  sram-senior-edificer
    
    A new directory (Decklists) will be created in the same folder as the running file, with nested directories of each commander listed in the "commanders.csv" file.
    Within each nested directory, a single csv will be created containing 300 unique GUIDs (deck IDs) from edhrec.
    Each csv is iterated through, first calling the edhrec API to gather the decklist.  The decklist is then passed to the scryfall API 
    in two requests (limit of 50 cards per request).  The response object is cleansed so we only save the columns listed in the "cols" variable.  
    This yields 300 csv files per commander, each containing a decklist and the corresponding card attributes.

COMMANDER ANALYSIS:
    This file utilizes the abilities file as well and the decklists gathered from Data Collection to create a power level both for commanders in general, as well as 
    for specific decks.  The abilities file and the "typerank" dictionary created are highly subjective and should be modified to fit your personal preferences.

CARD ANALYSIS:
    The "rulings.json" file was downloaded and renamed from the scryfall website under the "Oracle Cards" link, here: https://scryfall.com/docs/api/bulk-data. 
    
    This file creates a new column "ability_score" which is a summation of the ability scores for the oracle text of each card.  Using the abilities file,
    if the text of the ability is found in the oracle text, the corresponding power is added to the ability score.  For example, if the string 'until end of turn' is found
    in the oracle text, 1 is added to the ability score.  
        ***NOTE***
        The abilities file was generated manually.  Utilizing both ngrams of the oracle text as well as my own knowledge of the game, I created the abilities file using my own
        judgement.  This is not meant to be official nor complete.  Other strings can and should be added, and corresponding power rankings adjusted as you see fit.
        **********
    Next, a dictionary (ability_index) is created to convert each word into an integer for machine learning models.
    The data is then split into train/test data 80/20.  
    The labels (ability_score) are normalized to values between 0 and 1 (value/max_value), then rounded to 0 or 1 with the rounding point at .5.  This number can be adjusted 
    higher/lower to be more/less strict on what is considered good/bad.  
    Next, a function is defined that will create a tensor from the data.  The tensor will be of shape Len(train_data) by Len(ability_index).  As of 09/27/2020, 
    this is 17537 x 5402.
    After vectorizing, the training data is split into training/validation data.
    Lastly, the data is sent through the model with the model achieving 99.53% accuracy and 0.65 loss.
    A second method adapted from this article: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ is much more simplistic but provides a 
    higher accuracy score.  This method utilizes keras's built in embedding which does not require creating a dictionary of word indices, the raw text can be run into the method.  
