import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

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

# Read data using pandas and visulaize data using matplotlib
class ReadData(CreateDIR):
    def __init__(self, folder):
        CreateDIR.__init__(self, folder='', project_id='')
    
    # Load csv file using pd.read_csv() function
    def load_data(self):
        dataset_path = self.dataset_path()
        csv_path = os.path.join(dataset_path, "housing.csv")
        return pd.read_csv(csv_path)
    
    # read data for pandas data_frame
    def pandas_method(self, df):
        print('First 5 rows: \n', df.head())
        print('Last 5 rows: \n', df.tail())
        print('No. of Columns: \n', df.columns)
        # Print all rows
        # print(df.to_string()) 
        df.info()
        print(df.describe())
        print('Ocean Proximity: \n',df['ocean_proximity'].value_counts())
    
    # visulazie data using hist() function from matplotlib
    # and save the plot as .png by calling the user created save_fig() function
    # the save_fig() calls the plt.savefig() function of matplotlib
    def matplotlib_method(self, df):
        df.hist(bins=50, figsize=(16,10))
        SaveDataAndImage.save_fig(self, "attribute_histogram_plots")
        plt.show()
        
# Add a column and compare train_test_split() and StratifiedShuffleSplit()
class AddCompare(ReadData):
    def __init__(self, folder):
        ReadData.__init__(self, folder)
    
    def add_column(self):
        df = self.load_data()
        # df['median_income'].hist()
        # plt.show()
        # cut() function is used to segment and sort data values into bins. 
        # This function is also useful for going from a continuous variable to a categorical variable. 
        # In this example, cut() convert median_income column to groups of 1 to 5 ranges. 
        df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
        print(df['income_cat'].head())
        # df['income_cat'].hist()
        # plt.show()
        self.df = df

    def split_data(self):
        df = self.df

        # Split data into train and test set using train_test_split()
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        self.train_set = train_set
        self.test_set = test_set

        # Split data into train and test set using StratifiedShuffleSplit based on income_cat column
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df, df["income_cat"]):
            self.strat_train_set = df.loc[train_index]
            self.strat_test_set = df.loc[test_index]

    # This funciont takes a dataframe parameter
    def income_cat_proportions(self, df):
        # count income_cat column and divide it by length of dataframe
        return df["income_cat"].value_counts() / len(df)
    
    # Compare train_test_split() & StratifiedShuffleSplit()
    def compare_proprtions(self):
        df = self.df
        strat_test_set = self.strat_test_set
        test_set = self.test_set
        # Create a dataframe and store the data in compare_props
        compare_props = pd.DataFrame({
            # call income_cat_proportions() and get proportion of entire dataset.
            "Overall": self.income_cat_proportions(df),
            # call income_cat_proportions() and get proportion of strat_test_set - StratifiedShuffleSplit()
            "Stratified": self.income_cat_proportions(strat_test_set),
            # call income_cat_proportions() and get proportion of test_set - train_test_split()
            "Random": self.income_cat_proportions(test_set),
        }).sort_index()
        # error between "Random" and "Overall" column 
        compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
        # error between "Stratified" and "Overall" column
        compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
        print(compare_props)

if __name__ == '__main__':
    print('1 - Fetch & Extract Data \n2 - Read Data & Visualize \n3 - Visualize In Depth')
    user_input = int(input('Enter number :'))

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    if user_input == 1:
        data = SaveDataAndImage(folder='housing', url=HOUSING_URL)
        data.fetch_and_extract_data()
    elif user_input == 2: 
        # create object of ReadData()
        data = ReadData(folder='housing')
        df = data.load_data()
        data.pandas_method(df)
        data.matplotlib_method(df)
    elif user_input == 3:
        # create object of AddCompare()
        vd = AddCompare(folder='housing')
        vd.add_column()
        vd.split_data()
        vd.compare_proprtions()

