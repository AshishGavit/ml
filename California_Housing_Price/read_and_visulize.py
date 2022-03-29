from download_save_data import * 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# Read data using pandas and visulaize data using matplotlib
class ReadData(DownloadAndSave):
    def __init__(self):
        DownloadAndSave.__init__(self, url='')

    # Load csv file using pd.read_csv() function
    def load_data(self, data_path):
        os.makedirs(data_path, exist_ok=True)
        csv_path = os.path.join(data_path, "housing.csv")
        return pd.read_csv(csv_path)
    
    # read data for pandas data_frame
    def pandas_method(self, data_path):
        df = self.load_data(data_path)
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
    def matplotlib_method(self, data_path, img_path):
        df = self.load_data(data_path)
        df.hist(bins=50, figsize=(16,10))
        self.save_fig(img_path, "attribute_histogram_plots")
        plt.show()

# Add a column and compare train_test_split() and StratifiedShuffleSplit()
class CompareProprtions(ReadData):
    def __init__(self):
        ReadData.__init__(self)
    
    def add_column(self, data_path):
        df = self.load_data(data_path)
        # df['median_income'].hist()
        # plt.show()
        # cut() function is used to segment and sort data values into bins.  
        df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
        print('Created new column "income_cat"')
        print(df.head())
        # df['income_cat'].hist()
        # plt.show()
        return df

    def split_data(self, data_path):
        df = self.add_column(data_path)
        # Split data into train and test set using train_test_split()
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        # Split data into train and test set using StratifiedShuffleSplit based on income_cat column
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(df, df["income_cat"]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]
        return (df, train_set, test_set, strat_train_set, strat_test_set)

    # This funciont takes a dataframe parameter
    # count income_cat column and divide it by length of dataframe
    def income_cat_proportions(self, df):
        return df["income_cat"].value_counts() / len(df) 
    
    # Compare train_test_split() & StratifiedShuffleSplit()
    def compare_proprtions(self, data_path):
        df, train_set, test_set, strat_train_set, strat_test_set = self.split_data(data_path)

        # Create a dataframe and store the data in compare_props
        compare_props = pd.DataFrame({
            "Overall": self.income_cat_proportions(df), # call income_cat_proportions() and get proportion of entire dataset.
            "Stratified": self.income_cat_proportions(strat_test_set), # get proportion of strat_test_set - StratifiedShuffleSplit()
            "Random": self.income_cat_proportions(test_set), # get proportion of test_set - train_test_split()
        }).sort_index()

        # error between "Random" and "Overall" column 
        compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
        # error between "Stratified" and "Overall" column
        compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
        print('')
        print(compare_props)
    
    # Remove the income_cat column from 
    def remove_column(self, data_path):
        df, train_set, test_set, strat_train_set, strat_test_set = self.split_data(data_path)
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        return strat_train_set, strat_test_set

# this class discovers and visulize correlation between columns
# use StratifiedShuffleSplit split model to split the data and use that data to discover and visulize
class DiscoverAndVisulize(CompareProprtions):
    def __init__(self, data_path):
        CompareProprtions.__init__(self)
        strat_train_set, strat_test_set = self.remove_column(data_path)
        housing = strat_train_set.copy()
        self.housing = housing
    
    # Plot a Scatterplot and save it as 'housing_prices_scatterplot.png'
    def visulaize_data(self, img_path):
        housing = self.housing
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            sharex=False)
        plt.legend()
        self.save_fig(img_path, "housing_prices_scatterplot")
        plt.show()
    
    # display the correlation using pandas
    def correlation(self, img_path):
        housing = self.housing
        # add three columns in our current dataset 'housing'
        housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        housing["population_per_household"]=housing["population"]/housing["households"]
        corr_matrix = housing.corr() # use corr() for creating a metrix of correlation for all columns
        print(corr_matrix["median_house_value"].sort_values(ascending=False))
        attributes = ["median_house_value", "median_income", "total_rooms",
                    "housing_median_age"]
        scatter_matrix(housing[attributes], figsize=(10, 7)) # visulize correlation using scatter_matrix
        self.save_fig(img_path, "scatter_matrix_plot") # call SaveDataAndImage.save_fig() method to save as img.
        plt.show()

if __name__ == '__main__':
    data_path = 'datasets\\housing\\'
    img_path = 'images\\housing_img\\'
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    loop = True
    while loop == True:
        print('1 - Fetch & Extract Data \n2 - Read Data & Visualize \n3 - Add and Compare\n4 - Discover And Visulize\n')
        try:
            user_input = input('Enter number : ')
            user_input = int(user_input)
            if user_input == 1:
                # create object of DownloadAndSave class
                data = DownloadAndSave(url=HOUSING_URL)
                data.fetch_and_extract_data(data_path)
                data.save_fig(img_path, '')
            elif user_input == 2: 
                # create object of ReadData()
                rd = ReadData()
                rd.pandas_method(data_path)
                rd.matplotlib_method(data_path, img_path)
            elif user_input == 3:
                # create object of AddCompare()
                cp = CompareProprtions()
                cp.compare_proprtions(data_path)
            elif user_input == 4:
                da = DiscoverAndVisulize(data_path)
                da.visulaize_data(img_path)
                da.correlation(img_path)
        except ValueError:
            user_input = str(user_input)
            if user_input == 'exit':
                loop = False
                print('Ending program...')
            else:
                print('Oops! That was not a valid number. Try again....\n')