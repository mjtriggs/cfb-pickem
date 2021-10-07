# Produce CFB Pick 'em Excel Sheets
# Author: Matt Triggs
# Date: 07/10/2021

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
from datetime import date
import logging

import pull_data

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

if __name__ == "__main__":

    kPlayers = 50

    logging.basicConfig(level=logging.INFO)

    # Pull Team Lines
    logging.info('Pulling Team Lines...')
    lines = pull_data.pull_team_lines()
    
    # Pulling Player Details
    logging.info('Pulling Player Details...')
    qbs = pull_data.pull_qbs()
    rbs = pull_data.pull_rbs()
    wrs = pull_data.pull_wrs()

    logging.warning('JACK IS BAD')
    # Prepare Output
    logging.info('Cleaning Data...')
    qbs_clean = qbs[['NAME','ABBRV','CMP','ATT','TD','INT','RTG']]
    rbs_clean = rbs[['NAME', 'ABBRV','ATT','YDS','TD']]
    wrs_clean = wrs[['NAME','ABBRV','REC','YDS','TD']]

    # Get high scoring and high line games
    logging.warning('HORNS DOWN')
    logging.info('Getting High-Potential Games...')

    # 15 largest favourites
    big_favs = lines.sort_values('LINE').head(15)

    # Top 10 highest O/U games
    hi_sc_games = lines.sort_values('O/U', ascending=False).head(20)

    logging.info('Matching to top players...')
    logging.warning('MAX SUCKS')
    # Join on QBs/RBs/WRs by the abbreviation?
    big_favs_qb = big_favs.merge(qbs_clean.head(kPlayers))
    big_favs_rb = big_favs.merge(rbs_clean.head(kPlayers))
    big_favs_wr = big_favs.merge(wrs_clean.head(kPlayers))

    hi_sc_qb = hi_sc_games.merge(qbs_clean.head(kPlayers))
    hi_sc_rb = hi_sc_games.merge(rbs_clean.head(kPlayers))
    hi_sc_wr = hi_sc_games.merge(wrs_clean.head(kPlayers))

    logging.info('Outputting Excel File...')
    # Export from Python to Excel
    todays_date = str(date.today())
    writer = pd.ExcelWriter('data/output-' + todays_date + '.xlsx', engine='xlsxwriter')

    # big favs
    big_favs_qb.to_excel(writer, sheet_name = 'favs_qb')
    big_favs_rb.to_excel(writer, sheet_name = 'favs_rb')
    big_favs_wr.to_excel(writer, sheet_name = 'favs_wr')

    # high scores
    hi_sc_qb.to_excel(writer, sheet_name = 'hi_scr_qb')
    hi_sc_rb.to_excel(writer, sheet_name = 'hi_scr_rb')
    hi_sc_wr.to_excel(writer, sheet_name = 'hi_scr_wr')

    # all data
    qbs_clean.to_excel(writer, sheet_name = 'base_qb')
    rbs_clean.to_excel(writer, sheet_name = 'base_rb')
    wrs_clean.to_excel(writer, sheet_name = 'base_wr')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    logging.warning('GIG EM AGS')
    logging.info('Script Complete. Check Excel File.')