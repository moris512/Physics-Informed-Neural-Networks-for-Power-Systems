#Swing Equation - Inference (Section III.B)
#Python 3.7.X
#TensorFlow 1.X
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub, nu):
        
        self.lb = lb #lower bound of the system input
        self.ub = ub #upper bound of the system input
        
        self.x = X[:,0:1] #system input (mechanical power)
        self.t = X[:,1:2] #system input (time)
        self.u = u #system rotor angle
                
        self.nu = nu
               

        self.layers = layers #layers of the Neural Network
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers) #NN parameters (weights, bias)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize parameters (Inertia and Damping)
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32) #Inertia.
        self.lambda_2 = tf.Variable([-6.0], dtype=tf.float32) #Damping.
        
        #Final time
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
                
        self.u_pred = self.net_u(self.x_tf, self.t_tf) #predicted system output (rotor angle).
        self.f_pred = self.net_f(self.x_tf, self.t_tf) #predicted system output differentiation (rotor frequency).

        #Definition of the Loss Function
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        #Optimizer to minimize the Loss Function
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, #maximum number of iterations.
                                                                           'maxfun': 50000, #maximum number of function evaluations.
                                                                           'maxcor': 50, #maximum number of variable metric corrections used to define the limited memory matrix.
                                                                           'maxls': 50, #maximum number of line seeach steps per iteration.
                                                                           'ftol' : 1.0 * np.finfo(float).eps}) #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.

    
        self.optimizer_Adam = tf.train.AdamOptimizer() #Adaptative Moment Estimation as optimizer (1st and 2nd Momentum Gradient Descent).
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) #Apply ADAM optimizer to the Loss Function.
        
        init = tf.global_variables_initializer() #variable initializer and value holder.
        self.sess.run(init) #run the variable initializer.

    #Initializing the Neural Network
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]]) #Xavier Initialization = Initialize the Weights so that the 
            #variance of the activations are the same in every layer.
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32) #Bias = 0.
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #Define the activation function
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y #finally returning the (activation fuction * weight) * (bias)
            
    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases) #Store the values of the solution
        return u
    
    def net_f(self, x, t):
        #Define the system parameters
        lambda_1 = self.lambda_1  #Inertia
        lambda_2 = self.lambda_2  #Damping

        u = self.net_u(x,t)

        #Differentiate the obtained solution u with gradients.
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        f = lambda_1*u_tt + lambda_2*u_t + self.nu*tf.math.sin(u) - x
        
        return f
    
    #Print structure to track the results Loss, Lambda 1, and Lambda 2.
    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))

    #Train the Neural Network    
    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Actual Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)

                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1, self.lambda_2],
                                loss_callback = self.callback)
        
    #Do the prediction of the system state 
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict) #Rotor Angle
        f_star = self.sess.run(self.f_pred, tf_dict) #Rotor Frequency
        
        return u_star, f_star

    
if __name__ == "__main__": 
     
    nu=0.2
    N_u=100 #Number of initial and boundary data.
    layers = [2, 30, 30, 30, 30, 30,  1] #Number of layers

    data = scipy.io.loadmat('../Data/swingEquation_identification.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

  
        
    
    lb=np.array([0.08 ,  0.        ]) #lower bound of [P,t]
    ub=np.array([0.18,  20.        ]) #upper bound of [P,t]
    
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################

    #Run the NN with Noiseless Data
    noise = 0.0            
             
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    start_time = time.time()

    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, nu)
    model.train(0)
    
    u_pred, f_pred = model.predict(X_star)
    elapsed = time.time() - start_time                
        
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    error_lambda_1 = np.abs(lambda_1_value - 0.25)/0.25*100 #Inertia 
    error_lambda_2 = np.abs(lambda_2_value - 0.15)/0.15*100 #Damping
    
    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))  