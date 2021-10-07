# Pull Data Functions
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
import logging
#from datetime import date

def pull_team_lines(espn_url = 'https://www.espn.com/college-football/lines', lookup_location = "lookup-school-abbrv.csv"):
    
    # get raw data
    logging.info('Getting Lines HTML from ESPN...')
    page = requests.get(espn_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="fittPageContainer") # found by inspecting html

    # extract text from list of values
    logging.info('Starting data frame Manipulation')
    output = [i.text for i in results.find_all("td", class_="Table__TD")]

    # split list into a dataframe shape
    rows = int(len(output)/5)
    table = pd.DataFrame(np.reshape(output, (rows, 5)), columns = ['TEAM', 'RECORD', 'RAW_LINE', 'ML', 'FPI'])

    # Change column types
    table = table.astype({"TEAM": str, "RECORD": str, "RAW_LINE": float, "ML": str, "FPI":str})

    # Add in an indicators for the game number (used to join later)
    game_id = []
    for i in range(rows):
        game_id.append(math.floor(i/2))
    table.insert(0, 'GAME_ID', game_id)

    # create favs and dogs tables to get the lines and o/u seperated
    logging.info('Creating favourites and underdogs tables...')
    favs = table[table['RAW_LINE'] < 0]
    dogs = table[table['RAW_LINE'] > 0]

    # favs have the Raw Line as the line - this is fine, we just need to rename
    favs = favs.rename(columns={"RAW_LINE": "LINE"})

    # for the dogs, rename column for the o/u
    dogs = dogs.rename(columns={'RAW_LINE':'O/U'})

    # create df with just the data to merge on (game_id and o/u or line)
    dogs_to_join = dogs.copy()[['GAME_ID', 'O/U']]
    favs_to_join = favs.copy()[['GAME_ID', 'LINE']]

    # merge
    dogs = dogs.merge(favs_to_join)
    dogs['LINE'] = dogs['LINE'] * -1
    favs = favs.merge(dogs_to_join)

    # concat
    full = pd.concat([dogs, favs],sort=True)[['GAME_ID','TEAM','RECORD','O/U','LINE']]
    full = full.sort_values('GAME_ID')

    # Add in School Reference
    logging.info('Adding in School Abbreviations')
    school_lookup = pd.read_csv(lookup_location)
    school_lookup['TEAM'] = school_lookup['SCHOOL'] + ' ' + school_lookup['NICKNAME']
    school_lookup.drop(['SCHOOL', 'NICKNAME'], axis=1,inplace=True)

    # merge abbreviations
    full_abbrv = full.merge(school_lookup, how='left')
    full_abbrv.head()

    # If this has any rows, then something hasn't matched
    if full_abbrv[full_abbrv['ABBRV'].isna()].empty:
        print(full_abbrv[full_abbrv['ABBRV'].isna()]['ABBRV'])
        logging.error('Abbreviation not found in lookup')
    else:
        logging.info('Team Lines dataset created.')
        return full_abbrv

def pull_qbs(espn_url='https://www.espn.co.uk/college-football/stats/player/_/view/offense/table/passing/sort/passingYards/dir/desc'):
    # get raw data
    logging.info('Getting QBs HTML from ESPN...')
    page = requests.get(espn_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="fittPageContainer")

    # Get output into a list format for manipulation
    logging.info('Manipulating HTML into a useful form...')
    raw_output = [i.text for i in results.find_all("td", class_="Table__TD")]

    logging.info('Starting QB Dataframe Manipulation')
    n_rows = 50

    # get qb names
    qbs = []
    for i in range(n_rows*2):
        if i % 2 != 0:
            qbs.append(raw_output[i])

    school = []
    qb_clean = []

    # split out the school from the surname of the QB
    # could probably use REGEX to improve this, but it's fine
    for qb in qbs:
        if qb[len(qb)-4].isupper():
            # 4 letter team abbrv
            qb_clean.append(qb[:len(qb)-4])
            school.append(qb[len(qb)-4:])
        elif qb[len(qb)-3].isupper():
            # 3 letter abbrv
            qb_clean.append(qb[:len(qb)-3])
            school.append(qb[len(qb)-3:])
        else:
            # 2 letter abbrv
            qb_clean.append(qb[:len(qb)-2])
            school.append(qb[len(qb)-2:])
            
    # get other stats
    raw_stats = raw_output[n_rows*2:]
    table = pd.DataFrame(np.reshape(raw_stats, (int(len(raw_stats)/11), 11)), 
                        columns = ['POS','CMP','ATT','CMP%','YDS','AVG','LNG','TD','INT','SACK','RTG'])
    table.insert(0, 'NAME', qb_clean)
    table.insert(1, 'ABBRV', school)

    logging.info('QB table completed.')
    return table

def pull_rbs(espn_url='https://www.espn.co.uk/college-football/stats/player/_/view/offense/stat/rushing/table/rushing/sort/rushingYards/dir/desc'):
# get raw data
    logging.info('Getting RBs HTML from ESPN...')
    page = requests.get(espn_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="fittPageContainer")

    # Get output into a list format for manipulation
    logging.info('Manipulating HTML into a useful form...')
    raw_output = [i.text for i in results.find_all("td", class_="Table__TD")]

    logging.info('Starting RB Dataframe Manipulation')
    n_rows = 50

    # get rb names
    players = []
    for i in range(n_rows*2):
        if i % 2 != 0:
            players.append(raw_output[i])

    school = []
    player_clean = []

    # split out the school from the surname of the QB
    # could probably use REGEX to improve this, but it's fine
    for p in players:
        if p[len(p)-4].isupper():
            # 4 letter team abbrv
            player_clean.append(p[:len(p)-4])
            school.append(p[len(p)-4:])
        elif p[len(p)-3].isupper():
            # 3 letter abbrv
            player_clean.append(p[:len(p)-3])
            school.append(p[len(p)-3:])
        else:
            # 2 letter abbrv
            player_clean.append(p[:len(p)-2])
            school.append(p[len(p)-2:])
            
    # get other stats
    raw_stats = raw_output[n_rows*2:]
    table = pd.DataFrame(np.reshape(raw_stats, (int(len(raw_stats)/6), 6)), 
                        columns = ['POS','ATT','YDS','AVG','LNG','TD'])
    table.insert(0, 'NAME', player_clean)
    table.insert(1, 'ABBRV', school)

    logging.info('RB table completed.')
    return table

def pull_wrs(espn_url='https://www.espn.co.uk/college-football/stats/player/_/view/offense/stat/receiving'):
# get raw data
    logging.info('Getting WRs HTML from ESPN...')
    page = requests.get(espn_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="fittPageContainer")

    # Get output into a list format for manipulation
    logging.info('Manipulating HTML into a useful form...')
    raw_output = [i.text for i in results.find_all("td", class_="Table__TD")]

    logging.info('Starting WR Dataframe Manipulation')
    n_rows = 50

    # get wr names
    players = []
    for i in range(n_rows*2):
        if i % 2 != 0:
            players.append(raw_output[i])

    school = []
    player_clean = []

    # split out the school from the surname of the QB
    # could probably use REGEX to improve this, but it's fine
    for p in players:
        if p[len(p)-4].isupper():
            # 4 letter team abbrv
            player_clean.append(p[:len(p)-4])
            school.append(p[len(p)-4:])
        elif p[len(p)-3].isupper():
            # 3 letter abbrv
            player_clean.append(p[:len(p)-3])
            school.append(p[len(p)-3:])
        else:
            # 2 letter abbrv
            player_clean.append(p[:len(p)-2])
            school.append(p[len(p)-2:])
            
    # get other stats
    raw_stats = raw_output[n_rows*2:]
    table = pd.DataFrame(np.reshape(raw_stats, (int(len(raw_stats)/6), 6)), 
                        columns = ['POS','REC','YDS','AVG','LNG','TD'])
    table.insert(0, 'NAME', player_clean)
    table.insert(1, 'ABBRV', school)

    logging.info('WR table completed.')
    return table