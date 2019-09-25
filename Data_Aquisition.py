import numpy as np

import numpy.matlib

import pandas as pd

################################################################################################################################################################

class Data_Preparation (object):
    
    # Class that takes the name of a CSV file called as Dataset_File_Name, creates a numpy matrix and return the statistics of the data set.
    
    def __init__(self, Dataset_File_Name, Data_Matrix):
        
        # Defining the class attribute        
        self.Dataset_File_Name = Dataset_File_Name # The name of the CSV file of the dataset
        self.Data_Matrix = Data_Matrix # The matrix that will store the dataset
        
    def Matrix_Construction (self):
        
        # This function reads the CSV file and stores the data in a numpy matrix
        self.Data_Matrix = pd.read_csv( self.Dataset_File_Name, header = 0 )
        # Data_Matrix.to_numpy()
        return (self.Data_Matrix)
        
    def Binary_Classification (self):
        
        # This function takes the name of an ordinal output and converts it into a binary target (step needed for the red wine dataset)
        #Output_Column = input ("Tape the header`s name of the ordinal output:")
        #Threshold = int(input ("Set the threshold for the positive classification (y=1 if value >= threshold):"))
        Output_Column = 'quality'
        Threshold = 6
        self.Data_Matrix[ 'Binary_Output' ] = self.Data_Matrix[ Output_Column ].apply(lambda x: 1 if x >= Threshold else 0)  #
        return (self.Data_Matrix)
    
    def Features_Statistics (self):
        
        # Gives the basic statistics (mean, max, min, ...) of features and the percent of positive (y=1) and negative (y=0) answers
        print(self.Data_Matrix.describe())
        
        Positive = self.Data_Matrix[self.Data_Matrix['Binary_Output'] == 1].count()['Binary_Output'] / self.Data_Matrix['Binary_Output'].count() * 100
        Negative = self.Data_Matrix[self.Data_Matrix['Binary_Output'] == 0].count()['Binary_Output'] / self.Data_Matrix['Binary_Output'].count() * 100

        print("The percentage of good wine is ", Positive, " %.")
        print("The percentage of bad wine is ", Negative, " %.")
        
    def Select_Features (self, array_of_features):
        
        # Selects the features to be used by the Binary Classification models
        Selected_Features = self.Data_Matrix.iloc[:, array_of_features] # Separates the wanted features
        Selected_Features['Unit_Column'] = 1.0 # Adds an extra column of values 1.0
        print('The matrix of selected features is:')
        print(Selected_Features)
        return (Selected_Features)
    
    def Separate_Labels (self):
        y_true_labels = self.Data_Matrix.iloc[:, -1]
        print('The vector of labels is:')
        print(y_true_labels)
        return (y_true_labels)

################################################################################################################################################################
  
class Logistic_Regression (object):
    
    # Class to fit the Logistic Regression model upon a features matrix X and a vector of true labels y  
    
    def __init__(self, X_training_data, y_training_labels, Learning_rate, LR_Weights, Num_Iter, Epsilon, Boundary):
        
        # Defining the class attribute        
        self.X_training_data = X_training_data
        self.y_training_labels = y_training_labels
        self.Learning_rate = Learning_rate
        self.LR_Weights = LR_Weights
        self. Num_Iter = Num_Iter
        self.Epsilon = Epsilon
        self.Boundary = Boundary
        
    def Sigmoid_Vector (self, X, w):
        a = np.dot(X , w) # Generates a n dimensional vector
        sigma = 1 / (1 + np.exp ( -a )) # Generates a n dimensional vector of probabilities
        return sigma
    
    def Cross_Entropy_Loss (self, Weights):
        
        # Function to calculate the value of the Cross Entropy loss Function (CE_Data)
        Prediction = self.Sigmoid_Vector( self.X_training_data, Weights)
        Term_1 = np.dot( self.y_training_labels, np.log ( Prediction )) # First term of the CE function
        Term_2 = np.dot(1 - self.y_training_labels, np.log( 1 - Prediction )) # Second term of the CE function
        CE_Value = - ( Term_1 + Term_2 )
        return CE_Value
    
    def Update (self, Weights):
        
        Updated_Weights = Weights
        c = self.Sigmoid_Vector(self.X_training_data, Updated_Weights)
        Derivative_Vector = np.dot(self.X_training_data, np.add(self.y_training_labels, np.negative(c)))
        Updated_Weights = np.add(Updated_Weights, self.Learning_rate * Derivative_Vector)
        return Updated_Weights
    
    def Gradient_Descent (self):
        
        # Loop based on number of iterations and epsilon factor
        Trained_Weights = self.LR_Weights
        # CE_History = np.array(1)
        for i in range (self.Num_Iter):
            Temp = Trained_Weights
            Trained_Weights = self.Update( Trained_Weights )
            #CE = self.Cross_Entropy_Loss( Trained_Weights )
            # CE_History = np.append( CE )
            if np.linalg.norm( Trained_Weights - Temp ) < self.Epsilon:
                break
        return Trained_Weights
        # CE_History
    
    def Predict (self, Non_Seen_Data, Final_Weights):
        
        # Based on the fitted weights and on the decision boundary threshold, this function returns a binary vector with the predicted classification
        
        Prob_Vector = self.Sigmoid_Vector( Non_Seen_Data , Final_Weights)
        Classification = np.where(Prob_Vector >= self.Boundary, 1, 0)
        return Classification

################################################################################################################################################################    
  
Red_Wine_file = 'winequality-red.csv'
Red_Wine_Matrix = pd.DataFrame()
Red_Wine_Data = Data_Preparation( Red_Wine_file, Red_Wine_Matrix)
print (Red_Wine_Data.Matrix_Construction())
print (Red_Wine_Data.Binary_Classification())
Features_Wanted = [1, 2, 3, 9]
X = Red_Wine_Data.Select_Features(Features_Wanted)
X1 = X.to_numpy()
y = Red_Wine_Data.Separate_Labels()
y1 = y.to_numpy
Iter = 1000
eps = 0.01
LearnRate = 1
w0 = np.zeros(5)

Model = Logistic_Regression(X1,y1,LearnRate,w0,Iter,eps,0.5)
w = Model.Gradient_Descent()

    
        
        