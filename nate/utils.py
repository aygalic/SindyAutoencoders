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
    
    z = np.zeros((nsamples, dim, dim))
    dz = np.zeros((nsamples, dim, dim))
    ddz = np.zeros((nsamples, dim, dim))
    
    #create images 
    for i in range(nsamples):
        z[i] = create_image(x[i])
        
    
    #generate first derivative (we could easily do this way more efficiently but hey)
    #i'm assuming the object starts from rest so that dz|0 = ddz|0 = 0
    for i in range(nsamples-2):
        dz[i+1] = (z[i+2]-z[i])/(2*dt) #centered differencing
    
    dz[-1] = -(z[-1]-z[-2])/dt #backwards scheme
            
        
    #second derivatives
    ddz[1] = (dz[1]-dz[0])/dt
    for i in range(nsamples-3):
        ddz[i+2] = (dz[i+3]-dz[i+1])/(2*dt)
    
    ddz[-1] = -(dz[-1]-dz[-2])/dt    
    
    return z,dz,ddz
    
    
    
    
def z_derivative(model, data):
    """
    main difference between this function and the one they have is that i use ALL the model weights 
    """
    x, dx = data #unpack 

    temp = x
    dz = dx
    for i in range(len(model.weights)//2):
        weight = model.weights[2*i]
        bias = model.weights[2*i+1]

        #add a switch case here or something for selecting the activation 

        temp = tf.matmul(temp, weight) + bias #pre-activation
        temp = tf.nn.relu(temp) #activation 

        dz = tf.multiply(tf.cast(temp>0, float), tf.matmul(dz, weight)) #derivative relu 
        #dz = tf.multiply(temp*(1-temp), tf.matmul(dz, weight)) #derivative sigmoid 

    return dz


