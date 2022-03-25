from california_housing_price import *

if __name__ == '__main__':
    loop = True
    while loop == True:
        print('1 - Fetch & Extract Data \n2 - Read Data & Visualize \n3 - Add and Compare\n4 - Discove and Visualize \n5 - Data Preparation')

        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        try:
            user_input = input('Enter number : ')
            user_input = int(user_input)
            if user_input == 1:
                SDI = SaveDataAndImage(folder='housing', url=HOUSING_URL)
                SDI.fetch_and_extract_data()
            elif user_input == 2: 
                # create object of ReadData()
                RD = ReadData(folder='housing')
                RD.pandas_method()
                RD.matplotlib_method()
            elif user_input == 3:
                # create object of AddCompare()
                AC = AddCompare(folder='housing')
                AC.split_data()
                AC.compare_proprtions()
            elif user_input == 4:
                # create object of DiscoverAndVisulize
                DAV = DiscoverAndVisulize(folder='housing')
                DAV.visulaize_data()
                DAV.correlation()
            elif user_input == 5:
                # create object of DataPreparation
                DP = DataPreparation(folder='housing')
                DP.null_value_with_imputer()
                DP.convert_categorical_feature()
        except ValueError:
            user_input = str(user_input)
            if user_input == 'exit':
                loop = False
                print('Ending program...')
            else:
                print('Oops! That was not a valid number. Try again....\n')