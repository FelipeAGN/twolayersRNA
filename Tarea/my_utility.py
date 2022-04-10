import pandas as pd
import numpy  as np

# Initialize weights
def iniW_ae(prev,next):    
    weight_enc = iniW(next,prev)
    weight_dec = iniW(prev,next)
    return(weight_enc,weight_dec)
    
# Initialize one-weight    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# STEP 1: Feed-forward of AE
def forward_ae(w_enc,w_dec, layer0):
    ## calculate activation of nodes in layer 1
    output_enc = np.dot(w_enc, layer0)
    hlayer_1 = act_sigmoid(output_enc)
    ## calculate activation of nodes in layer 2
    output_dec = np.dot(w_dec, hlayer_1)
    hlayer_2 = act_sigmoid(output_dec)
    return(hlayer_1, hlayer_2) 

#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))  

# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward
def gradW_ae(layer0, hlayer_1, hlayer_2, w_enc, w_dec):    
    """d2 = (a2 - a0) * deriva_sigmoid(a2)
    gw2 = np.dot(d2,a1.transpose())
    d1 = np.dot(w2.transpose(),d2) * deriva_sigmoid(a1)
    gw1 = np.dot(d1,a0.transpose())"""
    
    #calculate gradient decoder
    delta_2 = np.cross(hlayer_2 - layer0, deriva_sigmoid(hlayer_2))
    gradient_2 = np.dot(delta_2, np.transpose(hlayer_1))
    
    # calculate gradient encoder
    w2T = np.transpose(w_dec)
    nose_a_que_le_llaman_esto = np.dot(w2T, delta_2)
    delta_1 = np.cross(nose_a_que_le_llaman_esto , deriva_sigmoid(hlayer_1))
    gradient_1 = np.dot(delta_1, np.transpose(layer0))
    return(gradient_1,gradient_2)     

# Update W of the AE
def updW_ae(w1,w2,lr,gw1,gw2):
    w2 = (w2 -lr) * (gw2*-1)
    w1 = (w1 -lr) * (gw1*-1)
    return(w1,w2)

"""
# Softmax's gradient
def grad_softmax(x,y,w,lambW):    
    ...    
    return(gW,Cost)
"""
# Calculate Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

"""
# Feed-forward of the DL
def forward_dl(x,W):        
    ...    
    return(zv)
"""    

"""
# Métrica
def metricas(...):
    ...    
    return()
"""   
#Confusion matrix
def confusion_matrix():    
    return()
#-----------------------------------------------------------------------
# Configuration of the DL 
#-----------------------------------------------------------------------
def load_config():      
    sae = []
    softmax = []
    datos_sae = np.genfromtxt("cnf_sae.csv", delimiter=',')
    datos_softmax =np.genfromtxt("cnf_softmax.csv", delimiter=',')
    
    sae.append(np.int32(datos_sae[0]))
    sae.append(np.float16(datos_sae[1]))
    sae.append(np.int32(datos_sae[2:]))
    

    
    softmax.append(np.int32(datos_softmax[0]))
    softmax.append(np.float16(datos_softmax[1]))
    softmax.append(np.float16(datos_softmax[2]))

    return(sae,softmax)

# Binary Label from raw data 

def Label_binary(x):
    return pd.get_dummies(x)
      

# Load data 
def load_data_csv(file_path):
    df = pd.read_csv(file_path,header = None)
    xe = df.iloc[:-1]
    x = df.iloc[-1,:]
    ye = Label_binary(x)
      
    return(xe,ye)
"""
# save weights of both SAE and Costo of Softmax
def save_w_dl():    
    ....
        
    
#load weight of the DL in numpy format
def load_w_dl():
    ...
    return()    

# save weights in numpy format
def save_w_npy():  
    ....
"""