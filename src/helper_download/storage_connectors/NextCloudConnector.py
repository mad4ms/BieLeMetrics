import owncloud
import json
import pandas as pd
from typing import Union, List, Optional
import os

# try:
#     from .StorageConnector import StorageConnector  # Import the base class
# except ImportError:
#     from StorageConnector import StorageConnector


class NextcloudConnector:
    def __init__(self, credentials: dict):
        self.client = self._init_owncloud_client(credentials)
        self.path_base = credentials["PATH_STORAGE_IN_NEXTCLOUD"]

    def _init_owncloud_client(self, credentials: dict) -> owncloud.Client:
        """
        Initialize and authenticate the Nextcloud client.
        """
        client = owncloud.Client(credentials["ENDPOINT_STORAGE_NEXTCLOUD"])
        client.login(
            credentials["USERNAME_STORAGE_NEXTCLOUD"],
            credentials["PASSWORD_STORAGE_NEXTCLOUD"],
        )
        return client

    def create_folders(self, base_path: str) -> None:
        """
        Create folders in Nextcloud.
        """
        folder_structure = [
            "/" + folder for folder in base_path.split("/") if folder
        ]
        for folder in folder_structure:
            try:
                self.client.mkdir(folder)
            except owncloud.HTTPResponseError as e:
                print(f"Failed to create folder {folder}: {e}")

    def download_file_by_game_id(
        self, game_id: int, extension: str = ".json", path_local: str = "."
    ) -> Optional[Union[dict, pd.DataFrame]]:
        """
        Download a file from Nextcloud by game ID.
        Returns the file as a JSON or a pandas DataFrame depending on the extension.
        """
        # List all files in the directory
        files = self.list_files()

        # Find the file with the game ID in the name
        for file in files:
            if (
                str(game_id) in file.get_name()
                and extension in file.get_name()
            ):
                # Check if the file already exists locally
                filename_local = file.get_name()
                local_file = os.path.join(path_local, filename_local)

                if not os.path.exists(local_file):
                    file_path = file.path
                    with open(local_file, "wb") as f:
                        self.client.get_file(file_path, f)
                else:
                    print(
                        f"File {local_file} already exists locally. Loading ..."
                    )
                # If the file exists locally, read it
                if extension == ".json":
                    with open(local_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                elif extension == ".csv":
                    return pd.read_csv(local_file)
                else:
                    print("Invalid file extension.")
                    return None

    def upload_file(
        self,
        filename: str,
        data: Union[pd.DataFrame, dict],
        path_remote: str,
        path_local: str,
    ) -> None:
        """
        Upload a file to Nextcloud.
        """
        full_path_remote = f"{path_remote}/{filename}"
        full_path_local = f"{path_local}/{filename}"
        if isinstance(data, pd.DataFrame):
            data.to_csv(full_path_local, index=False)
        else:
            with open(full_path_local, "w") as f:
                json.dump(data, f)

        self.client.put_file(full_path_remote, full_path_local)

    def list_files(self, depth: int = 5) -> List[owncloud.FileInfo]:
        """
        List files in the Nextcloud directory.
        """
        return self.client.list(self.path_base, depth=depth)

    def check_file_exists(self, file_name: str) -> bool:
        """
        Check if a file exists in Nextcloud.
        """
        try:
            files = self.list_files()
        except owncloud.HTTPResponseError as e:
            print(f"Error listing files: {e}")
            return False

        for file in files:
            if str(file_name) in file.get_name():
                print(f"File {file_name} exists in Nextcloud.")
                return True

        print(f"File {file_name} does not exist in Nextcloud.")
        return False
