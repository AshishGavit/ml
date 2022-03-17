import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Create Directories
class CreateDIR:
    def __init__(self, folder='', project_id=''):
        self.folder = folder
        self.project_id = project_id

    # check and create folder in datasets
    def dataset_path(self):
        FOLDER_PATH = os.path.join("datasets", self.folder)
        os.makedirs(FOLDER_PATH, exist_ok=True)
        return FOLDER_PATH

    # check and create folder image
    def image_path(self):
        IMAGES_PATH = os.path.join('.', "images", self.project_id)
        os.makedirs(IMAGES_PATH, exist_ok=True)
        return IMAGES_PATH

# downaloda, extract file and same images of California Housing Price
class SaveDataAndImage(CreateDIR):
    def __init__(self, folder, url):
        CreateDIR.__init__(self, folder='', project_id='')
        self.url = url        

    # download and extract .tgz file
    def fetch_and_extract_data(self):
        dataset_path = self.dataset_path()
        url = self.url
        tgz_path = os.path.join(dataset_path, "housing.tgz")
        urllib.request.urlretrieve(url, tgz_path)
        dataset_tgz = tarfile.open(tgz_path)
        dataset_tgz.extractall(path=dataset_path)
        dataset_tgz.close()

    # save matplotlib images as .png
    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        image_path = self.image_path()
        path = os.path.join(image_path, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

class ReadData(CreateDIR):
    def __init__(self, folder):
        CreateDIR.__init__(self, folder='', project_id='')

    def load_data(self):
        dataset_path = self.dataset_path()
        csv_path = os.path.join(dataset_path, "housing.csv")
        return pd.read_csv(csv_path)
    
    def pandas_method(self, df):
        print('First 5 rows: \n', df.head())
        print('Last 5 rows: \n', df.tail())
        print('No. of Columns: \n', df.columns)
        # Print all rows
        # print(df.to_string()) 
        df.info()
        print(df.describe())
        print('Ocean Proximity: \n',df['ocean_proximity'].value_counts())
    
    def matplotlib_method(self, df):
        df.hist(bins=50, figsize=(20,15))
        SaveDataAndImage.save_fig(self, "attribute_histogram_plots")
        plt.show()

if __name__ == '__main__':
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    data = SaveDataAndImage(folder='housing', url=HOUSING_URL)
    data.fetch_and_extract_data() 
    data = ReadData(folder='housing')
    df = data.load_data()
    data.pandas_method(df)
    data.matplotlib_method(df)
