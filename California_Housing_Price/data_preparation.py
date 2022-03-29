import pandas as pd
from read_and_visulize import CompareProprtions
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class DataPreparation(CompareProprtions):
    def __init__(self, data_path):
        CompareProprtions.__init__(self)
        strat_train_set, strat_test_set = self.remove_column(data_path)
        self.housing = strat_train_set.drop("median_house_value", axis=1) # add all columns but drop median_house_value for training purpose
        self.housing_labels = strat_train_set["median_house_value"].copy() # copy only median_housing_value column

    # fill null vlaue of one column at a time using fillna() method
    def null_value(self):
        housing = self.housing
        # find all rows which has null value from all columns in dataset
        sample_incomplete_rows = housing[housing.isnull().any(axis=1)].sort_index()
        print('Find null value in dataset')
        print(sample_incomplete_rows.to_string())
        # replace the null rows with median() value
        median = housing["total_bedrooms"].median() # get median of total_bedrooms columns
        sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # use fillna() to replace null value with median
        print('Replace null value using fillna() method by median of specific columns')
        print(sample_incomplete_rows)

    # find all null from entire dataset. This dataset should contain only interger value and not string value
    def null_value_with_imputer(self):
        housing = self.housing
        # Remove the text attribute because SimpleImputer can only calculated on numerical attributes
        housing_num = housing.drop('ocean_proximity', axis=1)
        # housing_num = housing.select_dtypes(include=[np.number])
        
        # create instance of SimpleImputer with median value
        imputer = SimpleImputer(strategy='median')
        imputer.fit(housing_num) # use fit() method to fit imputer instance to all columns of housing_num
        
        print('Statistics median value using imputer.statistics_')
        print(imputer.statistics_) # use statistics_ attribut of SimpleImputer() method to get median house value of all columns
        print('Statistics median value using housing.median().values')
        print(housing_num.median().values) # Check median housing all columns by using median() method and 

        #Transform the training set:
        X = imputer.transform(housing_num) # The result of transform() is a plain Numpy array
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index) # put X (plain Numpy array) into the dataframe
        print(housing_tr['total_bedrooms'][4629])
    
    # convert string to number or categorical values represented as a number
    def convert_categorical_feature(self):
        housing = self.housing
        housing_cat = housing[['ocean_proximity']] # get 'ocean_proximit' column
        housing_cat.head(10)

        # use OrdianlEncoder() method to convert categorial feature to 0 and 1
        ordinal_encoder = OrdinalEncoder() # create an object/instance of OrdinalEncoder() method
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) # use fit_transform() method to first fit and then transform the entire data
        print('ocean_proximity processed using OrdinalEncoder()')
        print(housing_cat_encoded[:10])

        print(ordinal_encoder.categories_) # use OrdinalEndor Attribut categories_ to view the category

        # OneHotEncoder() method convert categorial into sparse array. To disable sparse use sparse=False
        cat_encoder = OneHotEncoder(sparse=False) # create an object/instance of OneHotEncoder() method
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # use fit_transform() method to first fit and then transform the entire data
        print('ocean_proximity processed using OneHotEncoder()')
        print(housing_cat_1hot)
        # By default, the OneHotEncoder class returns a sparse array, but we can convert it to a dense array if needed by calling the toarray() method:
        # print(housing_cat_1hot.toarray())

if __name__ == '__main__':
    data_path = 'datasets\\housing\\'
    img_path = 'images\\housing_img\\'
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    loop = True
    while loop == True:
        print('1 - Fetch & Extract Data \n2 - Read Data & Visualize \n3 - Add and Compare\n4 - Discover And Visulize\n5 - Data Preparation\n')
        try:
            user_input = input('Enter number : ')
            user_input = int(user_input)
            if user_input == 1:
                from download_save_data import *
                # create object of DownloadAndSave class
                data = DownloadAndSave(url=HOUSING_URL)
                data.fetch_and_extract_data(data_path)
                data.save_fig(img_path, '')
            elif user_input == 2:
                from read_and_visulize import ReadData 
                # create object of ReadData()
                rd = ReadData()
                rd.pandas_method(data_path)
                rd.matplotlib_method(data_path, img_path)
            elif user_input == 3:
                from read_and_visulize import CompareProprtions
                # create object of AddCompare()
                cp = CompareProprtions()
                cp.compare_proprtions(data_path)
            elif user_input == 4:
                from read_and_visulize import DiscoverAndVisulize
                da = DiscoverAndVisulize()
                da.visulaize_data(data_path, img_path)
                da.correlation(data_path, img_path)
            elif user_input == 5:
                dp = DataPreparation(data_path)
                dp.null_value()
                dp.null_value_with_imputer()
                dp.convert_categorical_feature()
        except ValueError:
            user_input = str(user_input)
            if user_input == 'exit':
                loop = False
                print('Ending program...')
            else:
                print('Oops! That was not a valid number. Try again....\n')