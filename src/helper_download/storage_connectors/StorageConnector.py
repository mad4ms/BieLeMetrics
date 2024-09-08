from abc import ABC, abstractmethod
from typing import Union, List, Optional
import pandas as pd


class StorageConnector(ABC):
    """
    Abstract base class for all storage connectors (Nextcloud, AWS, Azure, etc.).
    """

    @abstractmethod
    def upload_file(
        self, filename: str, data: Union[pd.DataFrame, dict], path: str
    ) -> None:
        """
        Upload a file to the storage service.
        """
        pass

    @abstractmethod
    def download_file(
        self, file_name: str, extension: str = ".json"
    ) -> Optional[Union[dict, pd.DataFrame]]:
        """
        Download a file from the storage service.
        """
        pass

    @abstractmethod
    def list_files(self, depth: int = 5) -> List:
        """
        List files in the storage service directory.
        """
        pass

    @abstractmethod
    def create_folders(self, base_path: str) -> None:
        """
        Create folders in the storage service if they do not exist.
        """
        pass

    @abstractmethod
    def check_file_exists(self, file_name: str) -> bool:
        """
        Check if a file exists in the storage service.
        """
        pass
