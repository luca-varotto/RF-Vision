import numpy as np

PT = [30.0*np.pi/180,0.0*np.pi/180]

R_0 = np.array([[0,-1,0],\
                [-1,0,0],   
                [0,0,-1]])
R_alpha = np.array([[np.cos(-PT[0]),0,np.sin(-PT[0])],\
                [0,1,0],\
                [-np.sin(-PT[0]),0,np.cos(-PT[0])]])
R_beta = np.array([[1,0,0],\
                [0,np.cos(-PT[1]), -np.sin(-PT[1])],    
                [0,np.sin(-PT[1]), np.cos(-PT[1])]])
R = np.linalg.multi_dot([R_beta,R_alpha,R_0])

center =np.array([ [10.0], [10.0], [7.0] ])

G = np.block([ [ R, -np.dot(R, center)  ], [np.zeros((1,3)), 1] ]) 

resolution_px = 300
px2mm = 0.25
f = 30
u0 = int(resolution_px/2)
v0 = int(resolution_px/2) 
K =  np.array([[f*(1/px2mm),0,u0],\
                [0,f*(1/px2mm),v0],\
                [0,0,1]])

Pi_0 = np.hstack( (np.eye(3) , np.zeros((3,1))))
P =  np.linalg.multi_dot([K, Pi_0 ,G ])
Q = P[:,:-1]
q = P[:,-1]

Q_inv = np.linalg.inv(Q)
c_I = np.array([[int(resolution_px/2)], \
                [int(resolution_px/2)], \
                [1]])

# print(q, "\n \n",center) 
# print(np.dot(Q,center) +q.reshape(-1,1))  
print(np.dot(Q_inv,c_I), "\n \n", R[:,-1])
