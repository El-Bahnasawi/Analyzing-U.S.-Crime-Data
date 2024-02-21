from  utils import read_file_to_dict

urls_wget = {
    'https://data.ojp.usdoj.gov/resource/gcuy-rt5g.csv?$limit=1000000': "personal_victimization.csv",
    'https://data.ojp.usdoj.gov/resource/r4j4-fdwx.csv?$limit=1000000': "personal_population.csv",
    'https://data.ojp.usdoj.gov/resource/ynf5-u8nk.csv?$limit=1000000': "georgia_recidivism.csv",
    'https://www.statefirearmlaws.org/sites/default/files/2020-07/DATABASE_0.xlsx': "firearm_laws.xlsx",
    'https://www.statefirearmlaws.org/sites/default/files/2020-07/codebook_0.xlsx': "firearm_book.xlsx"
}

urls_gdown = {
    "https://docs.google.com/uc?export=download&id=1tS2kWpxKTwXlLbAgCdl_Ps9MbS3TWRpR": "population_states_1991_2021.csv",
    "https://docs.google.com/uc?export=download&id=1hVSLGoU1rzCUYQaTdfILptaqc8593Xqq": "offensecountperstate.csv"
}


code = read_file_to_dict(filename="txt_files/states_code.txt", flag=True)

pers_popdf_new_cols_dict = read_file_to_dict(filename="txt_files/pers_popdf_new_cols_dict.txt", flag=False)
pers_vectimdf_new_cols_dict = read_file_to_dict(filename="txt_files/pers_vectimdf_new_cols_dict.txt", flag=False)

category_dict = read_file_to_dict(filename="txt_files/category_dict.txt", flag=False)