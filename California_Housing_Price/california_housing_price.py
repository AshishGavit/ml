import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

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
        print(FOLDER_PATH)
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
        self.folder = folder
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
    def pandas_method(self):
        df = self.load_data()
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
    def matplotlib_method(self):
        df = self.load_data()
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
        return df

    def split_data(self):
        df = self.add_column()

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
        df = self.add_column()
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

# this class discovers and visulize correlation between columns 
class DiscoverAndVisulize(AddCompare):
    def __init__(self, folder):
        AddCompare.__init__(self, folder)
    
    # Remove the income_cat column
    def remove_column(self):
        self.split_data()
        strat_train_set = self.strat_train_set
        strat_test_set = self.strat_test_set
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        return strat_train_set, strat_test_set
    
    # Plot a Scatterplot and save it as 'housing_prices_scatterplot.png'
    def visulaize_data(self):
        strat_train_set, strat_test_set = self.remove_column()
        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            sharex=False)
        plt.legend()
        SaveDataAndImage.save_fig(self, "housing_prices_scatterplot")
        plt.show()
    
    # display the correlation using pandas
    def correlation(self):
        strat_train_set, strat_test_set = self.remove_column()
        housing = strat_train_set.copy()
        # add three columns in our current dataset 'housing'
        housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        housing["population_per_household"]=housing["population"]/housing["households"]
        corr_matrix = housing.corr() # use corr() for creating a metrix of correlation for all columns
        print(corr_matrix["median_house_value"].sort_values(ascending=False))
        attributes = ["median_house_value", "median_income", "total_rooms",
                    "housing_median_age"]
        scatter_matrix(housing[attributes], figsize=(10, 7)) # visulize correlation using scatter_matrix
        SaveDataAndImage.save_fig(self,"scatter_matrix_plot") # call SaveDataAndImage.save_fig() method to save as img.
        plt.show()

class DataPreparation(DiscoverAndVisulize):
    def __init__(self, folder):
        DiscoverAndVisulize.__init__(self, folder)

    def null_value(self):
        strat_train_set, strat_test_set = self.remove_column()
        housing = strat_train_set.drop("median_house_value", axis=1) # add all columns but drop median_house_value for training purpose
        housing_labels = strat_train_set["median_house_value"].copy() # copy only median_housing_value column
        # find all rows which has null value from all columns in dataset
        sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
        print(sample_incomplete_rows)
        # replace the null rows with median() value
        median = housing["total_bedrooms"].median() # get median of total_bedrooms columns
        sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # use fillna() to replace null value with median
        print(sample_incomplete_rows)
        return strat_train_set, strat_test_set

    def null_value_with_imputer(self):
        strat_train_set, strat_test_set = self.null_value()
        housing = strat_train_set.drop("median_house_value", axis=1) # add all columns but drop median_house_value for training purpose
        housing_labels = strat_train_set["median_house_value"].copy() # copy only median_housing_value column
        
        # Remove the text attribute because SimpleImputer can only calculated on numerical attributes
        housing_num = housing.drop('ocean_proximity', axis=1)
        # housing_num = housing.select_dtypes(include=[np.number])
        
        # create instance of SimpleImputer with median value
        imputer = SimpleImputer(strategy='median')
        imputer.fit(housing_num) # use fit() method to fit imputer instance to all columns of housing_num
        print(imputer.statistics_) # use statistics_ attribut of SimpleImputer() method to get median house value of all columns

        print(housing_num.median().values) # Check median housing all columns

        #Transform the training set:
        X = imputer.transform(housing_num) # The result of transform() is a plain Numpy array
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index) # put X (plain Numpy array) into the dataframe

        print(housing_tr)
    
    def convert_categorical_feature(self):
        strat_train_set, strat_test_set = self.null_value()
        housing = strat_train_set.drop("median_house_value", axis=1) # add all columns but drop median_house_value for training purpose
        housing_labels = strat_train_set["median_house_value"].copy() # copy only median_housing_value column

        housing_cat = housing[['ocean_proximity']] # get 'ocean_proximit' column
        housing_cat.head(10)

        # use OrdianlEncoder() method to convert categorial feature to 0 and 1
        ordinal_encoder = OrdinalEncoder() # create an object/instance of OrdinalEncoder() method
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) # use fit_transform() method to first fit and then transform the entire data
        print(housing_cat_encoded[:10])

        print(ordinal_encoder.categories_) # use OrdinalEndor Attribut categories_ to view the category

        # OneHotEncoder() method convert categorial into sparse array. To disable sparse use sparse=False
        cat_encoder = OneHotEncoder(sparse=False) # create an object/instance of OneHotEncoder() method
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # use fit_transform() method to first fit and then transform the entire data
        print(housing_cat_1hot)
        # By default, the OneHotEncoder class returns a sparse array, but we can convert it to a dense array if needed by calling the toarray() method:
        # print(housing_cat_1hot.toarray())