import os


def create_directory(directory):
    exists = os.path.isdir(directory)
    if( exists == False ):
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)
        else:
            print ("Successfully created the directory %s " % directory)
    else:
        print ("The directory %s already exists" % directory)