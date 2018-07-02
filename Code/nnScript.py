import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import pickle
import time

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);  # epsilon is 
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return 1.0/(1.0 + np.exp(-z)) 
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat_obj = loadmat('mnist_all.mat') 
    train_data_temp = np.zeros((0,784))  
    train_label_temp = np.zeros((0,1),dtype=np.int64) 
    test_data = np.zeros((0,784)) 
    test_label = np.zeros((0,1),dtype=np.int64)
    for i in range(10): 
        mat_get_curr = mat_obj.get('train'+str(i));
        train_data_temp = np.concatenate((train_data_temp,mat_get_curr))
        num_get_curr = mat_get_curr.shape[0]
        temp = np.zeros((num_get_curr,1))
        temp.fill(i)
        train_label_temp = np.concatenate((train_label_temp,temp))
        mat_get_curr = mat_obj.get('test'+str(i));
        test_data = np.concatenate((test_data,mat_get_curr))
        num_get_curr = mat_get_curr.shape[0]
        temp = np.zeros((num_get_curr,1))
        temp.fill(i)
        test_label = np.concatenate((test_label,temp)) 
        
    equal_features = np.all(train_data_temp == train_data_temp[0,:],axis=0)  
    
    same_cols = np.where(equal_features == True)  
    train_data_temp = np.delete(train_data_temp,same_cols[0],axis=1)  
    
    train_data = np.zeros((50000,train_data_temp.shape[1])) 
    train_label = np.zeros((50000,1),dtype=np.int64) 
   
    s = random.sample(range(train_data_temp.shape[0]),60000)
    for i in range(50000):
        train_data[i,:] = train_data_temp[s[i],:]
        train_label[i,0] = train_label_temp[s[i],0]
        
    validation_data = np.zeros((10000,train_data_temp.shape[1])) 
    validation_label = np.zeros((10000,1),dtype=np.int64)  
    for j in range(50000,60000):
        validation_data[j-50000,:] = train_data_temp[s[j],:]
        validation_label[j-50000,0] = train_label_temp[s[j],0]  
    
    train_data = train_data / 255.0 

    validation_data = validation_data / 255.0 
       
    test_data = test_data / 255.0 
    
    test_data = np.delete(test_data,same_cols[0],axis=1)
                                            
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))      
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))      
    obj_val = 0.0 

    neg_log_likelihood_error = 0.0
    w1_gradiant = np.zeros(w1.shape,dtype=np.float64)
    w2_gradiant = np.zeros(w2.shape,dtype=np.float64)
    
    b_node = np.ones((training_data.shape[0],1), dtype=np.float64)
    training_data = np.append(training_data,b_node,axis=1)

    input_to_hidden =  np.dot(training_data, w1.transpose())                   
            
    input_to_hidden = sigmoid(input_to_hidden)
        
    hidden_bias = np.ones((training_data.shape[0],1), dtype=np.float64)
    input_to_hidden = np.append(input_to_hidden,hidden_bias,axis=1)  
        
    hidden_to_output = np.dot(input_to_hidden, w2.transpose());
                    
    hidden_to_output = sigmoid(hidden_to_output)   
           
    x = np.copy(hidden_to_output)
 
    x.fill(0.0)
    for n in range(training_data.shape[0]):    
        x[n, training_label[n][0]] = 1.0    
        
    neg_log_likelihood_error = np.sum((x * np.log(hidden_to_output)) + ((1.0-x)*np.log(1.0-hidden_to_output)))        
        
    w1_gradiant_temp = np.dot((hidden_to_output-x),w2[:,0:n_hidden])
        
    w1_gradiant += np.dot(((1- input_to_hidden[:,0:n_hidden]) * input_to_hidden[:,0:n_hidden] * w1_gradiant_temp).transpose() , training_data)
   
    w2_gradiant = np.dot(( hidden_to_output - x ).transpose(), input_to_hidden)
                
    obj_val = (-1.0/training_data.shape[0]) * neg_log_likelihood_error

    reg_term =   np.sum(np.square(w1)) + np.sum(np.square(w2))

    obj_val = obj_val + ((lambdaval/(2*training_data.shape[0]))*reg_term)   

    w1_gradiant = np.add(w1_gradiant , (lambdaval*w1))/training_data.shape[0]
    w2_gradiant  = np.add(w2_gradiant , (lambdaval*w2))/training_data.shape[0]
    obj_grad = np.concatenate((w1_gradiant.flatten(), w2_gradiant.flatten()),0)
    
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.zeros((data.shape[0],1))
    temp = np.ones((data.shape[0],1), dtype=np.float64)
    data = np.append(data,temp,axis=1)
    for n in range(data.shape[0]): 
        
        hidden_to_output =  np.dot(w1 , data[n][:].transpose())                    
            
        hidden_to_output = np.append(sigmoid(hidden_to_output),[1],axis=0)

        output_layer = np.dot(w2 , hidden_to_output);

        labels[n][0] = np.argmax(sigmoid(output_layer))
        
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.4;


args = (n_input, n_hidden, n_class, train_data[0:50000], train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 100}    # Preferred value.

start_time = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
print("---Training Time : %s seconds ---" % (time.time() - start_time))

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
pickle.dump((n_input,n_hidden,w1,w2,lambdaval) , open( "params_hidden20.pickle", "wb" ))

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')