from helper_preprocessing.GameDataPreprocessing import GameDataPreprocessing
import argparse

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
    game_data_preprocessing.create_kinexon_snippets_from_events()
    game_data_preprocessing.process_game_events()


def main():
    # get gameid from args
    parser = argparse.ArgumentParser()
    parser.add_argument("path_file_sportradar", help="path to sportradar file")
    parser.add_argument("path_file_kinexon", help="path to kinexon file")
    args = parser.parse_args()

    path_file_kinexon = args.path_file_kinexon
    path_file_sportradar = args.path_file_sportradar

    # path_file_sportradar = "./data/raw/2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_sportradar.json"
    # path_file_kinexon = "./data/raw/2023-08-24_gd_01_id_42307421_HCErlangen_vs_TSVHannover-Burgdorf_kinexon.csv"

    process_game(
        path_file_sportradar,
        path_file_kinexon,
    )


if __name__ == "__main__":
    main()
