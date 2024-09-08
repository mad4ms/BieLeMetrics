from helper_preprocessing.GameDataPreprocessing import GameDataPreprocessing


# example is:
# path_file_sportradar = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
# path_file_kinexon = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

# game_data_preprocessing = GameDataPreprocessing(
#     path_file_sportradar, path_file_kinexon
# )
# game_data_preprocessing.clean_game_data()
# game_data_preprocessing.extract_game_data_to_events()


def process_game(path_file_sportradar: str, path_file_kinexon: str) -> None:
    """
    Process a game by cleaning and extracting data.
    """
    game_data_preprocessing = GameDataPreprocessing(
        path_file_sportradar, path_file_kinexon
    )
    game_data_preprocessing.clean_game_data()
    game_data_preprocessing.extract_game_data_to_events()


if __name__ == "__main__":

    path_file_sportradar = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    path_file_kinexon = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

    process_game(
        path_file_sportradar,
        path_file_kinexon,
    )
