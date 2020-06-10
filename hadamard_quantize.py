# Copyright 2020 Department of Electrical Engineering, Stanford University. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is a package for locally private data transmission. 


#%matplotlib inline
import RR_RAPPOR
import math
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
from functions import *


#the Hadamard randamized responce when \epsilon < 1
class Hadamard_Quantize:
    def __init__(self, d, eps, comm=1, encode_acc = 0, proj = False): # absz: alphabet size, pri_para: privacy parameter
        #set encode_acc = 1 to enable fast encoding by intializing hadamard matrix
        self.d = d #input alphabet size d
        self.D = int(math.pow(2,math.ceil(math.log(d,2)))) #padding d to power of 2
        self.k = min(comm, math.ceil(eps*math.log(math.e,2)), math.ceil(math.log(d,2))+1) # choose k^* as min(k, log(e^\epsilon))
        self.B = int(self.D/math.pow(2, self.k-1)) # set block size
        self.rr = RR_RAPPOR.Randomized_Response(int(math.pow(2, self.k)), eps)
        self.eps = eps
        print("D:%d k:%d B:%d"%(self.D, self.k, self.B))
        
        self.ifencode_acc = encode_acc #whether to accelerate encoding process
        if encode_acc == 1:
            self.H_D = scipy.linalg.hadamard(self.D) # initialize Hadarmard matrix

        self.proj = proj # If proj = true, project estimated distribution to the probability simplex
    
                                  
    def encode_symbol(self, idx, in_symbol):  # encode a single symbol into a privatized version
        
        # j: group index of node idx/ X: local observation
        j = int(idx % self.B)
        X = int(in_symbol)
        
        sign = Hadamard_entry(self.D, j, X)
        loc = int(X/self.B)

        # Compressed message: first k bits for location/ last bit for sign
        loc_sign = 2*loc 
        if Hadamard_entry(self.D, j, X) == -1:
            loc_sign = loc_sign+1

        loc_sign_priv = self.rr.encode_string(np.array([loc_sign]))

        return loc_sign_priv[0]
    '''    
    def encode_string_fast(self,in_list):  # encode string into a privatized string
        # Vectorize "encode_symbol" 
        l = len(in_list)
        n = int(l/self.B)*self.B
        group_list = np.array(in_list[:n]).reshape(self.B, int(n/self.B))
        for g_idx in range(self.B):
            g_count,g_axis = np.histogram(group_list[g_idx], range(self.D+1)) 

        loc_sign = np.zeros(n)
        for idx in range(n):
            j = int(idx % self.B)
            X = int(in_list[idx])
            loc_sign[idx] = 2*int(X/self.B)
            if Hadamard_entry(self.D, j, X) == -1:
                loc_sign[idx] = loc_sign[idx]+1

        out_list = self.rr.encode_string(np.array(loc_sign))

        return out_list
    '''
    def encode_string(self,in_list):  # encode string into a privatized string
        # Vectorize "encode_symbol" 
        n = len(in_list)
        loc_sign = np.zeros(n)
        for idx in range(n):
            j = int(idx % self.B)
            X = int(in_list[idx])
            loc_sign[idx] = 2*int(X/self.B)
            if Hadamard_entry(self.D, j, X) == -1:
                loc_sign[idx] = loc_sign[idx]+1

        out_list = self.rr.encode_string(np.array(loc_sign))

        return out_list

    def decode_string(self, out_list): # get the privatized string and learn the original distribution

        l = len(out_list)
        n = int(l/self.B)*self.B
        # Create B histograms, each histogram specifies the emprirical distribution of each group
        histograms = np.zeros((self.B, int(self.D/self.B))) 
        for i in range(n):
            loc = int(out_list[i]/2) # location info in the first k bits
            sign = (-1)**(out_list[i]%2) # sign info in the last bit
            histograms[i%self.B][loc] += sign*(self.B/n)*(math.exp(self.eps)+math.pow(2, self.k)-1)/(math.exp(self.eps)-1) # Normalized
        #print(histograms)
        # Obtain estimator of q
        q = np.zeros((self.B, int(self.D/self.B)))
        for j in range(self.B):
            q[j,:] = FWHT_A(int(self.D/self.B), histograms[j,:])
        
        q = q.reshape((self.D,), order = 'F')

        # Perform inverse Hadamard transform to get p
        p_D = FWHT_A(self.D, q)/self.D
        p = p_D[:self.d]

        # TODO: Project p onto d-dim probability simplex to reduce MSE
        #if self.proj:
        #p = SimplexProj(p)

        return p

    def decode_string_fast(self, out_list, normalization = 0): # get the privatized string and learn the original distribution

        l = len(out_list)
        n = int(l/self.B)*self.B
        group_list = np.array(out_list[:n]).reshape(int(n/self.B), self.B).T
        #group_list = np.array(out_list[:n]).reshape(self.B, int(n/self.B))

        # Create B histograms, each histogram specifies the emprirical distribution of each group
        histograms = np.zeros((self.B, int(self.D/self.B))) 
        for g_idx in range(self.B):
            g_count,g_axis = np.histogram(group_list[g_idx], range(int(math.pow(2, self.k))+1)) 
            histograms[g_idx] = g_count[::2] - g_count[1::2]
            histograms[g_idx] = histograms[g_idx]*(math.exp(self.eps)+math.pow(2, self.k)-1)/(math.exp(self.eps)-1)*(self.B/n)
        #print(histograms)
        # Obtain estimator of q
        q = np.zeros((self.B, int(self.D/self.B)))
        for j in range(self.B):
            q[j,:] = FWHT_A(int(self.D/self.B), histograms[j,:])
        
        q = q.reshape((self.D,), order = 'F')

        # Perform inverse Hadamard transform to get p
        p_D = FWHT_A(self.D, q)/self.D
        p = p_D[:self.d]

        if normalization == 0: 
            dist = probability_normalize(p) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(p) #simplex projection

        return dist

