import argparse
import requests
import zipfile
import os

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    # First attempt: request the download page
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Gets the confirmation token from Google Drive for large files."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save the file chunk by chunk."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)

def main(urls, output_dir):
    downloaded_files = []  # To keep track of downloaded file paths
    for url in urls:
        file_id = url.split('id=')[1].split('&')[0]  # Extract the file ID
        file_name = f"{file_id}.download"  # Set a default name for the downloaded file
        destination = os.path.join(output_dir, file_name)
        print(f"Start downloading {url}")
        download_file_from_google_drive(file_id, destination)
        downloaded_files.append(destination)  # Add downloaded file path to the list
        print(f"Finished downloading {file_name}")
    
    return downloaded_files  # Return the list of downloaded file paths

def combine_files_into_zip(file_paths, output_zip):
    """Create a ZIP file with all the downloaded files."""
    with zipfile.ZipFile(output_zip + "/combined_files.zip", 'w') as zipf:
        for file in file_paths:
            # Add each file to the ZIP archive
            zipf.write(file, os.path.basename(file))
    print(f"Files combined into {output_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Google Drive and combine them into a ZIP file.")
    parser.add_argument(
        "urls", 
        nargs="*", 
        default=[
            "https://drive.google.com/uc?id=1aPU9G7wqCAkhAcOR5AeuYtoPyQ108WCM&export=download",
            "https://drive.google.com/uc?id=1coc76dCF1BXh3tED3XOB9JstRfiI_Q6O&export=download"
        ], 
        help="List of Google Drive URLs to download files from. Defaults to two example URLs."
    )
    parser.add_argument("--output_dir", default="downloads", help="Directory to save downloaded files (default: downloads).")
    parser.add_argument("--output_zip", default="output", help="Name of the output ZIP file (default: combined_files.zip).")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_zip):
        os.makedirs(args.output_zip)

    # Download files and get their paths
    downloaded_files = main(args.urls, args.output_dir)

    # Combine all downloaded files into a zip
    combine_files_into_zip(downloaded_files, args.output_zip)
