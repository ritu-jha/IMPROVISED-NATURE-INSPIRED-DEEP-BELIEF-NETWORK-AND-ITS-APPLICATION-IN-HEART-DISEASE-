from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from accuracy_score import accuracy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import functools
from pyswarm import pso
import csv
import random


data=[]
def Deep_belief_network(X,*args):
	x,y,regularization_nn,learning_rate_RBM,learning_rate_nn,n_iter_RBM,batch_size_RBM,batch_size_nn,n_iter_nn=args
	n_components_RBM,n_components_nn_=X
	rbm_model_1=BernoulliRBM(n_components=n_components_RBM,n_iter=n_iter_RBM,learning_rate=learning_rate_RBM,batch_size=batch_size_RBM,verbose=0)
	#rbm_model_2=BernoulliRBM(n_components=n_components_RBM[1],n_iter=n_iter_RBM,learning_rate=learning_rate_RBM,batch_size=batch_size_RBM,verbose=0)
	#rbm_model_3=BernoulliRBM(n_components=n_components_RBM[2],n_iter=n_iter_RBM,learning_rate=learning_rate_RBM,batch_size=batch_size_RBM,verbose=0)
	
	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)
	min_max_scaler = MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	rbm_train=rbm_model_1.fit_transform(X_train)
	#X_train=rbm_model_2.fit_transform(X_train)
	#X_train=rbm_model_3.fit_transform(X_train)
	nn_model=MLPClassifier(activation='relu', algorithm='sgd', alpha=regularization_nn,
       batch_size=batch_size_nn, hidden_layer_sizes=(n_components_nn_), learning_rate='constant',
       learning_rate_init=learning_rate_nn, max_iter=n_iter_nn,shuffle=True, validation_fraction=0.1, verbose=False,random_state=1)
	nn_model.fit(rbm_train,Y_train)
	X_test=min_max_scaler.fit_transform(X_test)
	X_test=rbm_model_1.transform(X_test)
	#X_test=rbm_model_2.transform(X_test)
	#X_test=rbm_model_3.transform(X_test)
	Y_pred=nn_model.predict(X_test)
	acc=accuracy(Y_test,Y_pred)
	f= open('data.txt', 'a') 
	f.write(str(n_components_RBM)+' '+str(n_components_nn_)+' '+str(acc)+'\n')
	f.close()	
	return -1.0*acc
def main():
	with open('heart_data_norm.txt', 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader, None)  # skip header
		rows = np.array([r for r in reader])
		random.shuffle(rows)  # randomly shuffle the data
	#print(rows)
	Y=rows[:,13]
	X=rows[:,1:13]
	X = X.astype(np.float32)
	regularization_nn=0
	#n_components_RBM=50
	#n_components_nn_=20
	learning_rate_RBM=0.006
	learning_rate_nn=0.1
	n_iter_RBM=20
	batch_size_RBM=100
	batch_size_nn=100
	n_iter_nn=5000
	args=X,Y,regularization_nn,learning_rate_RBM,learning_rate_nn,n_iter_RBM,batch_size_RBM,batch_size_nn,n_iter_nn
	#Deep_belief_network(X,Y,regularization_nn,n_components_RBM,n_components_nn_,learning_rate_RBM,learning_rate_nn,n_iter_RBM,batch_size_RBM,batch_size_nn,n_iter_nn)
	
	'''
	cost_func = functools.partial(Deep_belief_network,X=X, Y=Y,n_components_RBM=100,n_components_nn_=128,learning_rate_RBM=0.06,learning_rate_nn=0.01,n_iter_RBM=200,
	batch_size_RBM=100,batch_size_nn=100,n_iter_nn=2000)
   	i=0
   	swarm=pso.ParticleSwarm(cost_func,dim=1,size=30)
   	best_scores=[(i,swarm.best_score)]
   	print(best_scores[-1])
   	while swarm.best_score>1e-9 and i<100 :
   		swarm.update()
   		i=i+1
   		if(swarm.best_score<best_scores[-1][1]):
   			best_scores.append((i,swarm.best_score))
   			print (best_scores[-1])
   	best_alpha=swarm.g
   	'''
   	
   	xopt,fopt=pso(Deep_belief_network,[6,15],[12,60],args=args,maxiter=30,debug=True,swarmsize=60,minstep=0)
	print(xopt,fopt)
	#acc=Deep_belief_network(x=0.00010482,args)
	print fopt
	
if __name__ == '__main__':
  main()	
