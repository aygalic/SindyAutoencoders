import numpy as np 
import tensorflow as tf 
from scipy.integrate import solve_ivp

def generate_spring_data(params, t, x0):
    spring_f = lambda t, x, params: [x[1], -params['c']*x[1]-(params['k']/params['m'])*x[0]]
    
    data = solve_ivp(lambda t,x: spring_f(t,x,params), 
                    t_span = (t[0], t[-1]), 
                    y0 = x0,
                    t_eval = t).y.T
    return data 


def spring_to_movie(x, dt=0.01, dim = 100):
    """
    naive implementation of the function provided in the pendulum example.  in the case where our images are large
    the discrepancy between the differencing scheme and the explicit derivatives should be neglibible. 
    
    maybe later we should compute these for realz
    
    todo:
    - explicit derivatives instead of differencing scheme
    - scale it up a bit to consider the number of initial conditions we're considering (like they did, batch info tf)
    - location and scale params for size and framing of the mass in frame 
    """
    
    #note the ordering on y (i'm guessing they did it so the top-down convention is reversed)
    y1,y2 = np.meshgrid(np.linspace(1.5,-1.5,dim),np.linspace(1.5,-1.5,dim)) 
    
    #defining a guassian over the image centered where our point mass is 
    create_image = lambda x : np.exp(-((y1 - x)**2 + (y2 - 0)**2)/.05)
    
    nsamples = x.shape[0]
    
    z = np.zeros((nsamples, dim, dim), dtype = 'float32')
    dz = np.zeros((nsamples, dim, dim), dtype = 'float32')
    ddz = np.zeros((nsamples, dim, dim), dtype = 'float32')
    
    #create images 
    for i in range(nsamples):
        z[i] = create_image(x[i])
        
    
    #generate first derivative (we could easily do this way more efficiently but hey)
    #i'm assuming the object starts from rest so that dz|0 = ddz|0 = 0
    for i in range(nsamples-2):
        dz[i+1] = (z[i+2]-z[i])/(2*dt) #centered differencing
    
    dz[-1] = -(z[-1]-z[-2])/dt #backwards scheme
            
        
    #second derivatives
    #ddz[1] = (dz[1]-dz[0])/dt
    for i in range(nsamples-3):
        ddz[i+2] = (dz[i+3]-dz[i+1])/(2*dt)
    
    ddz[-1] = -(dz[-1]-dz[-2])/dt    
    
    return z,dz,ddz



def z_derivative(model, data, activations):
    assert len(model.weights)//2 == len(activations)
    
    x, dx = data #unpack 
    
    #cast rn 
    x = x.astype('float32')
    dx = dx.astype('float32')

    z = x
    dz = dx
    for i in range(len(model.weights)//2):
        if activations[i] == 'relu':
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1] #pre-activation
            dz = tf.multiply(tf.cast(z>0, float), tf.matmul(dz, model.weights[2*i])) #derivative relu 

            z = tf.nn.relu(z) #relu
            
        elif activations[i] == 'sigmoid':
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1] #pre-activation
            dz = tf.multiply(z*(1-z), tf.matmul(dz, model.weights[2*i])) #derivative sigmoid 

            z = tf.nn.sigmoid(z) #sigmoid
            
        else: #linear
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1]
            dz = tf.matmul(dz, model.weights[2*i])

    return dz
    
    
    
    

#i'm trusting the authors had this one figured out 
def z_derivative2(model, data, activations):
    assert len(model.weights)//2 == len(activations)
    
    x, dx, ddx = data #unpack 
    
    #need this for matrix multiplications rn 
    #x = x.astype('float32')
    #dx = dx.astype('float32')
    #ddx = ddx.astype('float32')
    
    """the following code is truncated to the relevant activation functions for us"""
    z = x
    dz = dx
    ddz = ddx
    
    # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
    for i in range(len(model.weights)//2):
        if activations[i] == 'relu':
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1]
            a_derivative = tf.cast(z>0, float)
            dz = tf.multiply(a_derivative, tf.matmul(dz, model.weights[2*i]))
            ddz = tf.multiply(a_derivative, tf.matmul(ddz, model.weights[2*i]))  #in relu case f'' = 0
            z = tf.nn.relu(z)
        
        elif activations[i] == 'sigmoid':
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1]
            z = tf.sigmoid(z)
            dz_prev = tf.matmul(dz, model.weights[2*i])
            
            #f' & f''
            sigmoid_derivative = tf.multiply(z, 1-z) 
            sigmoid_derivative2 = tf.multiply(sigmoid_derivative, 1 - 2*z)

            #chain rule 
            dz = tf.multiply(sigmoid_derivative, dz_prev)
            ddz = tf.multiply(sigmoid_derivative2, tf.square(dz_prev)) \
                  + tf.multiply(sigmoid_derivative, tf.matmul(ddz, model.weights[2*i]))
            
        else: #linear
            z = tf.matmul(z, model.weights[2*i]) + model.weights[2*i+1]
            dz = tf.matmul(dz, model.weights[2*i])
            ddz = tf.matmul(ddz, model.weights[2*i])
            
              
    return dz, ddz


