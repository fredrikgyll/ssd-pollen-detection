import shutil
from pathlib import Path

import requests


class CDNConnector:

    # constructor
    def __init__(self, api_key, storage_zone, storage_zone_region="de"):
        """
        creates an object for using bunnyCDN \n
        api_key=Your Bunny Storage ApiKey/FTP key \n
        storage_zone=Name of your storage zone \n
        """

        self.headers = {"AccessKey": api_key}

        if storage_zone_region == "de" or storage_zone_region == "":
            self.base_url = f"https://storage.bunnycdn.com/{storage_zone}/"

        else:
            self.base_url = (
                f"https://{storage_zone_region}.storage.bunnycdn.com/{storage_zone}/"
            )

    def get_storaged_objects(self, cdn_path):
        """
        returns files and folders stored information stored in CDN (json data)\n
        path=folder path in cdn\n
        """
        request_url = self.base_url + cdn_path

        if cdn_path[-1] != "/":
            request_url = request_url + "/"

        response = requests.get(request_url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file(self, cdn_path, download_path=None):
        """Download file from your cdn storage
        cdn_path: storage path for the file (including file name) in cdn,
            use / as seperator e.g. 'images/logo.png'
        download_path: (default=None, stores in your present working directory)
            pass your desired download path with file name,
            will rewrite already existing files, if do not exists create them.

        Note, directory will not be created
        """

        request_url = self.base_url + cdn_path
        with requests.get(request_url, headers=self.headers, stream=True) as r:
            if r.status_code == 404:
                raise ValueError("No such file exists")

            if r.status_code != 200:
                raise Exception("Some error, please check all settings once and retry")

            download_path = download_path or Path(cdn_path).name

            with open(download_path, "wb") as file:
                shutil.copyfileobj(r.raw, file)

    def upload_file(self, cdn_path, file_path, file_name=None):
        """
        uploads your files to cdn server \n
        cdn_path - directory to save in CDN \n
        file_name - name to save with cdn \n
        file_path - locally stored file path,
        if none it will look for file in present working directory
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f'Could not find local file: {file_path}')

        file_name = file_name or file_path.name
        cdn_path = cdn_path.rstrip("/")

        request_url = f'{self.base_url}{cdn_path}/{file_name}'
        s = requests.Session()
        s.headers.update(self.headers)

        with file_path.open("rb") as file:
            response = s.put(request_url, data=file)

        response.raise_for_status()

        return response.json()

    def remove(self, cdn_dir):
        """
        deletes a directory or file from cdn \n
        cdn_dir=complete path including file on CDN \n
        for directory make sure that path ends with /
        """
        request_url = self.base_url + cdn_dir
        response = requests.delete(request_url, headers=self.headers)
        response.raise_for_status()
        return response.json()
