# mtg-cardanalysis
Analyze oracle text of Magic: The Gathering cards and create a model to predict power level

DATA COLLECTION:
    This file reads in a list of commanders from a csv file.  The commander names need to be in the format expected by edhrec.  For example:  sram-senior-edificer
    A new directory (Decklists) will be created in the same folder as the running file, with nested directories of each commander listed in the "commanders.csv" file.
    Within each nested directory, a single csv will be created containing 300 unique GUIDs (deck IDs) from edhrec.
    Each csv is iterated through, first calling the edhrec API to gather the decklist.  The decklist is then passed to the scryfall API 
    in two requests (limit of 50 cards per request).  The response object is cleansed so we only save the columns listed in the "cols" variable.  
    This yields 300 csv files per commander, each containing a decklist and the corresponding card attributes.
