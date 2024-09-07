import urllib.request

def download_file(url, destination):
    """
    Download a file from a URL and save it to a destination, displaying the download progress.

    Args:
        url (str): The URL of the file to download.
        destination (str): The local file path to save the downloaded file.

    Raises:
        urllib.error.URLError: If there is an error downloading the file.
    """
    def _download_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"Downloading: {percent}%", end="\r")

    try:
        urllib.request.urlretrieve(url, destination, reporthook=_download_progress)
        print("Download complete.")
    except urllib.error.URLError as e:
        print(f"Error downloading file: {e}")
        raise
