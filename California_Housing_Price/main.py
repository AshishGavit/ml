from data_preparation import *

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
                da = DiscoverAndVisulize(data_path)
                da.visulaize_data(img_path)
                da.correlation(img_path)
            elif user_input == 5:
                dp = DataPreparation()
                dp.null_value(data_path)
        except ValueError:
            user_input = str(user_input)
            if user_input == 'exit':
                loop = False
                print('Ending program...')
            else:
                print('Oops! That was not a valid number. Try again....\n')