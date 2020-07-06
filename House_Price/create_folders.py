import os
import shutil

def create_folders(folder_name='empty_folder'):

    if os.path.isdir(folder_name):
            #shutil.rmtree(folder_name)
            #os.mkdir(folder_name) 
            pass
    else:
        os.mkdir(f"./{folder_name}") 
