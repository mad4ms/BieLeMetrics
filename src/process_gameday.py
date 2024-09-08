from helper_preprocessing.GameDataPreprocessing import GameDataPreprocessing

from glob import glob

# example is:
# path_file_sportradar = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
# path_file_kinexon = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

# game_data_preprocessing = GameDataPreprocessing(
#     path_file_sportradar, path_file_kinexon
# )
# game_data_preprocessing.clean_game_data()
# game_data_preprocessing.extract_game_data_to_events()


def process_gameday(gameday: int, path_to_files: str) -> None:
    """
    Process a gameday by extracting data list.
    """
    list_files_sportradar = glob(
        f"{path_to_files}/*sportradar.json", recursive=True
    )
    list_files_kinexon = glob(f"{path_to_files}/*kinexon.csv", recursive=True)

    for file_sportradar, file_kinexon in zip(
        list_files_sportradar, list_files_kinexon
    ):
        game_data_preprocessing = GameDataPreprocessing(
            file_sportradar, file_kinexon
        )
        game_data_preprocessing.clean_game_data()
        game_data_preprocessing.extract_game_data_to_events()


if __name__ == "__main__":

    path_file_sportradar = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    path_file_kinexon = "2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

    process_gameday(1, "data/raw")
