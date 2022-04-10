# Deep-Learning:Training via BP+GD

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
"""
def train_softmax(x,y,param):
    .....        
    return(w,costo)
"""
# AE's Training 
def train_ae(layer0, capa, u, iteraciones):
    
    # layer0 = 256 caracteristicas, capa = 192 nodos
    w_enc , w_dec = ut.iniW_ae(layer0.shape[0], capa) 
    print('shape weights layer0 to hlayer1: ', w_enc.shape)
    print('shape weights hlayer1 to hlayer2: ', w_dec.shape)           
    
    for i in range(iteraciones):
        
        ## activate layers ##
        hlayer_1,hlayer_2 = ut.forward_ae(w_enc, w_dec, layer0)
        print('encoder nodes shape: ', hlayer_1.shape)
        print('decoder nodes shape: ', hlayer_2.shape)
        
        ## calculate gradients ##
        gradient_1, gradient_2 = ut.gradW_ae(layer0, hlayer_1, hlayer_2, w_enc, w_dec)
        print('shape gradient 1: ', gradient_1.shape)
        print('shape gradient 2: ', gradient_2.shape)
        
        ## update weights ##
        w_enc, w_dec = ut.updW_ae(w_enc, w_dec, u, gradient_1, gradient_2)
    return(w_enc)

#SAE's Training 
def train_sae(layer0, p_sae):
    W = []
    for capa in p_sae[2]:
        w_enc = train_ae(layer0, capa, p_sae[1], p_sae[0])
        W.append(w_enc)
        activation_hlayer1 = np.dot(w_enc, layer0)
        layer0 = ut.act_sigmoid(activation_hlayer1)
    return(W, layer0) 
   
# Beginning ...
def main():
    p_sae, p_sft = ut.load_config()           
    xe, ye       = ut.load_data_csv('dtrain.csv')
    W, Xr        = train_sae(xe, p_sae)         
    #Ws, cost    = train_softmax(Xr,ye,p_sft)
    #ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

