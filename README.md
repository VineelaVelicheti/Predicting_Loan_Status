# Predicting_Loan_Status

Steps for implementation:

Unzip the submission folder "012445186_Assignment".

********Folders Description **************

When unzipped the Source Code folder consists of Src, Images along with read_me file and the csv file of dataset.

1. Source_Code - consists of all the python scripts in .py form.
2. Images - This is the folder used to save all images from the python scripts.
	    You can create a new folder or you can change the path to the images folder provided in the submission folder.
	    However these images already exist in the provided folder.
  
************Package Installation*********

Install the required Packages:

1. Anaconda for Jupiter notebook
2. Pandas
3. Numpy
4. Matplotlib for Visualizations
5. SQLite (optional)

**************Data Exploration***********

Load the file "Data_Exploration.py" in Anaconda - Juptyer Notebook.

Change the file path at the end of the code to the "Images" folder for exporting the target image in its ".svg" format.
Run the Data Exploration file to obtain the loan data table to the "mortgage" database created in SQLite.

***************Preprocessing*************

Load the "Preprocessing.py" in Anaconda - Jupyter Notebook
Run the file to obtain new tables (X_train, X_test, Y_train, Y_test) in the SQLite Database. 

****************Classification***********

For Classification 3 files namely: Prediction_PCA.py, Prediction_SVD.py, Prediction_Variance_Threshold.py are provided.
Each file follows different Dimensionality Reduction Technique but same Classification Techniques.

The instructions mentioned below can be applied commonly for all these files:

1. Load the file in Anaconda - Jupyter Notebook
2. Run the file stepwise to see how the performance measures vary for each algorithm.
3. Change the file path at the end of the code to the "Images" folder for exporting various images in "svg" format to the folder "Images".
4. Experimental Evaluations can be seen through the images saved in this folder.
5. The top 10 Features affecting the model are also displayed.

**********End-of-file*************


