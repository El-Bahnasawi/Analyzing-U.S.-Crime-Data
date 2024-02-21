import wget
import gdown
import os
import concurrent.futures
from constants import *
import pandas as pd

class DirectoryManager:
    def __init__(self, directory_name):
        self.directory_name = directory_name
        self.path = os.path.join(os.getcwd(), self.directory_name)
        self.files = ["personal_victimization.csv", "personal_population.csv", "georgia_recidivism.csv", 
                      "firearm_laws.xlsx", "firearm_book.xlsx", "population_states_1991_2021.csv", "offensecountperstate.csv"]

    def create_directory(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def check_files(self):
        dir_list = os.listdir(self.path)
        number_of_files_missing = 0
        for file in self.files:
            if file not in dir_list:
                print(file)
                number_of_files_missing += 1
        return number_of_files_missing

class FileDownloader:
    def __init__(self, directory_manager):
        self.directory_manager = directory_manager

    def download_wget(self, url_output_pair):
        url, output = url_output_pair
        output = os.path.join(self.directory_manager.path, output)
        wget.download(url, output)

    def download_gdown(self, url_output_pair):
        url, output = url_output_pair
        output = os.path.join(self.directory_manager.path, output)
        gdown.download(url, output)

    def download_files(self, urls_wget, urls_gdown):
        self.directory_manager.create_directory()
        number_of_files_missing = self.directory_manager.check_files()

        if number_of_files_missing == len(self.directory_manager.files):
            # Download using wget concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.download_wget, urls_wget.items())

            # Download using gdown concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.download_gdown, urls_gdown.items())
        elif number_of_files_missing == 0:
            print("All files are already downloaded")


def write_dict_to_file(filename: str, dict: dict):
    with open(filename, "w") as f:
        for key, value in dict.items():
            f.write(key + " " + value + "\n")
        f.close()

def read_file_to_dict(filename: str, flag: bool) -> dict[str, str]:
    g = []
    with open(filename, "r") as f: 
        g = f.readlines()

    dict_data = {}
    for line in g:
        temp = line.split()
        if len(temp) > 1:
            if flag == True:
                key, value = ' '.join(temp[:-1]), temp[-1]
            else:
                key, value = temp[0], ' '.join(temp[1:])
            dict_data[key] = value
    return dict_data


def offense_count_df_handling(offensecountdf: pd.DataFrame, category_dict: dict, code: dict) -> pd.DataFrame:
    # Replace offense names using the category_dict
    offensecountdf['offenseCategory'] = offensecountdf['offenseName'].replace(category_dict)

    # Map state codes
    offensecountdf['Code'] = offensecountdf['stateName'].map(code)

    # Drop unnecessary column
    offensecountdf.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

    return offensecountdf


import json

def load_mappings_from_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        mappings = json.load(file)

    # Convert keys to integers
    mappings = {col: {int(key): value for key, value in mapping.items()} for col, mapping in mappings.items()}

    return mappings

def handle_dataframe(df: pd.DataFrame, mappings_file: str, new_cols_dict: dict) -> pd.DataFrame:
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Rename columns
    df_copy.rename(columns=new_cols_dict, inplace=True, errors="raise")

    # Load mappings from JSON file
    column_mappings = load_mappings_from_json(mappings_file)

    # Apply all mappings
    for column, mapping in column_mappings.items():
        df_copy[column] = df_copy[column].map(mapping)

    return df_copy
