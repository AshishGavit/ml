from california_housing_price import *

if __name__ == '__main__':
    loop = True
    while loop == True:
        print('1 - Fetch & Extract Data \n2 - Read Data & Visualize \n3 - Visualize In Depth\n4 - Discove and Visualize')

        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        try:
            user_input = input('Enter number : ')
            user_input = int(user_input)
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
                vd.split_data()
                vd.compare_proprtions()
            elif user_input == 4:
                # create object of 
                view = DiscoverAndVisulize(folder='housing')
                view.remove_column()
        except ValueError:
            user_input = str(user_input)
            if user_input == 'exit':
                loop = False
                print('Ending program...')
            else:
                print('Oops! That was not a valid number. Try again....')