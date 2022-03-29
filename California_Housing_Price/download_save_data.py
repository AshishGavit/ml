import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt

class DownloadAndSave:
    def __init__(self, url):
        self.url = url

    # download and extract .tgz file
    def fetch_and_extract_data(self, data_path):
        print('Downloading...')
        os.makedirs(data_path, exist_ok=True)
        url = self.url
        tgz_path = os.path.join(data_path, "housing.tgz")
        urllib.request.urlretrieve(url, tgz_path)
        dataset_tgz = tarfile.open(tgz_path)
        dataset_tgz.extractall(path=data_path)
        dataset_tgz.close()
        print('Done')
    
    # save matplotlib images as .png 
    def save_fig(self, img_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(img_path, exist_ok=True)
        if fig_id != '':
            path = os.path.join(img_path, fig_id + "." + fig_extension)
            print("Saving figure", fig_id)
            if tight_layout:
                plt.tight_layout()
            plt.savefig(path, format=fig_extension, dpi=resolution)
        else:
            print('No image available!')


if __name__ == '__main__':
    data_path = 'datasets\\housing\\'
    img_path = 'images\\housing_img\\'
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    url = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fig_id = "attribute_histogram_plots"

    data = DownloadAndSave(url=url)
    data.fetch_and_extract_data(data_path)
    data.save_fig(img_path, '')