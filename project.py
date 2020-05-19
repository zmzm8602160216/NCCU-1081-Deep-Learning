#!python
#!/usr/bin/env python
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import math 
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import random
#------------------------------------------------------------------------
undesire_=0
class0_=[] #Record LTS y in class 0
class1_=[] #Record LTS y in class 1
class0_index=[] #Record LTS rows in class 0
class1_index=[] #Record LTS rows in class 1
X_call=0 #For the use of callback function to process on sample x
Y_call=0 #For the use of callback function to process on sample y
LTS_X_ID=0 #Record LTS rows
epp=0 #Record how many epochs have been run after callbacks
Layer1W=0 #Record weight after softening for hidden layer
Layer2W=0 #Record weight after softening for output layer
integ=0 #For judging integrating
Loss_inte=0 #Record loss after weight tuning during integrating mechanism 
savesoft1=0 #Record weight before softening
savesoft2=0
SaveweightH=0 #Record weight before weight tuning
SaveweightO=0
def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 
def Mymodel(reg,node):
	initial=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
	regulizer=keras.regularizers.l2(l=reg) #l2 norm
	optimizer=keras.optimizers.SGD(learning_rate=0.05,momentum=0.05,nesterov=True) #SGD optimizer
	model=keras.Sequential()
	#Relu in hidden layer, linear in output layer
	model.add(tf.keras.layers.Dense(units=node,input_dim=8, name="hiddenLayer",activation='relu',use_bias=True,bias_initializer=initial,kernel_initializer=initial,dynamic=True,kernel_regularizer=regulizer,bias_regularizer=regulizer))	
	model.add(tf.keras.layers.Dense(units=1, name="outputLayer",activation='linear',use_bias=True,bias_initializer=initial,kernel_initializer=initial,dynamic=True,kernel_regularizer=regulizer,bias_regularizer=regulizer))	 
	model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

	return model
#------------------------------------------------------------------------
class myCallback(tf.keras.callbacks.Callback):
	def __init__(self,logs={}):
		super(myCallback, self).__init__()
		self.class0=[] #Store prediction of samples in each class
		self.class1=[]
		self.class0Index=[] 
		self.class1Index=[]
		self.predict=[]	
		self.undesire=0 #Record whether undesired attractor has happened
		self.condition=0 #Judge condition L
		self.ep=0 #Record epoch
		self.inte=0 #For judging success of integrating
		self.threshold=0.00001 #threshold for undesired attractor
	def on_epoch_begin(self, epoch, logs={}):
		self.class0.clear()
		self.class1.clear()
		self.class0Index.clear()
		self.class1Index.clear()
	def on_epoch_end(self, epoch, logs={}):
		#Predict the samples during weight tuning
		self.predict=np.array(self.model.predict(X_call)).reshape(-1,1)
		for i in range(self.predict.shape[0]):
			if(self.predict[i][0]>0.5):
				self.class1.append(Y_call[i][0])
				self.class1Index.append(LTS_X_ID[i][0])
			else:
				self.class0.append(Y_call[i][0])
				self.class0Index.append(LTS_X_ID[i][0])	
			
		optimizer = self.model.optimizer
		lr=optimizer.learning_rate
		mo=optimizer.momentum
		#Judge condition L
		if((len(self.class1)!=0) and (len(self.class0)!=0)):	
			self.alfa=np.amin(np.array(self.class1)) 
			self.beta=np.amax(np.array(self.class0))
			if(self.alfa>self.beta):
				self.condition=1
			else:
				self.condition=0
		else:
			self.condition=0

		if(epoch==0):
			self.previous=logs.get('loss')
			self.ep=epoch+1
			if(self.condition==1):
				self.undesire=0
				self.inte=1
				print(self.ep)
				print('initial condition satisfy')
				info(self.undesire,self.class0,self.class1,self.class0Index,self.class1Index,self.inte,self.previous,self.ep)
				self.model.stop_training = True #condition L is satisfied
			else:
				self.model.layers[0].set_weights(SaveweightH)
				self.model.layers[1].set_weights(SaveweightO)
				self.weightH=self.model.layers[0].get_weights()		
				self.weightO=self.model.layers[1].get_weights()
				
		else:
			self.new=logs.get('loss')

			if(self.condition==1):
				self.undesire=0
				self.inte=1
				self.ep=epoch+1
				print(self.ep)
				print('condition satisfy') 
				info(self.undesire,self.class0,self.class1,self.class0Index,self.class1Index,self.inte,self.new,self.ep)	
				self.model.stop_training = True #condition L is satisfied
			else:	
				if(self.new<self.previous):
						tf.keras.backend.set_value(self.model.optimizer.learning_rate,lr*1.2) #Increase learning rate
						tf.keras.backend.set_value(self.model.optimizer.momentum,mo*1.2)
						
						self.weightH=self.model.layers[0].get_weights()		#Save best weight so far
						self.weightO=self.model.layers[1].get_weights()		
						self.ep=epoch+1
						print(self.ep)
						print('keep moving')
						if(self.ep==500):
							self.undesire=1 #Treat it as undesired attractor if it runs to the end
							self.inte=0
							info(self.undesire,self.class0,self.class1,self.class0Index,self.class1Index,self.inte,self.new,self.ep)
							
						
				else:
					if((lr>self.threshold) or (mo>self.threshold)):
						tf.keras.backend.set_value(self.model.optimizer.learning_rate,lr*0.7) #Reduce learning rate
						tf.keras.backend.set_value(self.model.optimizer.momentum,mo*0.7) #Reduce momentum
						self.model.layers[0].set_weights(self.weightH) #Set up as the best weight so far
						self.model.layers[1].set_weights(self.weightO)
						self.ep=epoch+1
						print(self.ep)
						print('weight tune waits')
						if(self.ep==500):
							self.undesire=1
							self.inte=0
							info(self.undesire,self.class0,self.class1,self.class0Index,self.class1Index,self.inte,self.new,self.ep)
					else:
						self.undesire=1 #For recording undesired attractor
						self.inte=0 #For recording not success of integrating
						self.ep=epoch+1 #Record epochs
						print(self.ep)
						print("undesired attractor")
						#print(lr.value())
						#print(mo.value())
						info(self.undesire,self.class0,self.class1,self.class0Index,self.class1Index,self.inte,self.new,self.ep)
						self.model.stop_training = True #undesired attractor

			self.previous=self.new
##########################################################################
class soften(tf.keras.callbacks.Callback):
	def __init__(self,logs={}):
		super(soften, self).__init__()
		self.class0=[]
		self.class1=[]
		self.class0Index=[]
		self.class1Index=[]
		self.predict=[]
		self.condition=0
		self.threshold=0.0001
	def on_epoch_begin(self, epoch, logs={}):
		self.class0.clear()
		self.class1.clear()
		self.class0Index.clear()
		self.class1Index.clear()
		self.condition=0			
	def on_epoch_end(self, epoch, logs={}):
		#Predict the result
		self.predict=np.array(self.model.predict(X_call)).reshape(-1,1)
		for i in range(self.predict.shape[0]):
			if(self.predict[i][0]>0.5):
				self.class1.append(Y_call[i][0])
				self.class1Index.append(LTS_X_ID[i][0])
			else:
				self.class0.append(Y_call[i][0])
				self.class0Index.append(LTS_X_ID[i][0])

		
		optimizer = self.model.optimizer
		lr=optimizer.learning_rate
		mo=optimizer.momentum
		#Judge condition L
		if((len(self.class1)!=0) and (len(self.class0)!=0)):	
			self.alfa=np.amin(np.array(self.class1)) 
			self.beta=np.amax(np.array(self.class0))
			if(self.alfa>self.beta):
				self.condition=1
			else:
				self.condition=0
		else:
			self.condition=0


		if(epoch==0):
			if(self.condition==0):
				info2(savesoft1,savesoft2)
				##
				print('initial soft not satisfy')
				##
				self.model.stop_training = True
			else:
				self.previous=logs.get('loss')
				self.weightH=self.model.layers[0].get_weights()		
				self.weightO=self.model.layers[1].get_weights()
				tf.keras.backend.set_value(self.model.optimizer.learning_rate,lr*1.2)
				tf.keras.backend.set_value(self.model.optimizer.momentum,mo*1.2)
				##
				print('initial soft success')
				##

		else:
			self.new=logs.get('loss')


			if(self.new<self.previous):
				if(self.condition==1):
					tf.keras.backend.set_value(self.model.optimizer.learning_rate,lr*1.2)
					tf.keras.backend.set_value(self.model.optimizer.momentum,mo*1.2)
					self.weightH=self.model.layers[0].get_weights()		
					self.weightO=self.model.layers[1].get_weights()
					if(epoch==999):
						print('soft to the end success1')
						info2(self.weightH,self.weightO)
					##
					print('soft con. satisfy')
					##
				else:
					info2(self.weightH,self.weightO)
					##
					print('soft con. not satisfy out')
					##
					self.model.stop_training = True 
				
			else:
				if((lr>self.threshold) or (mo>self.threshold)):	
					tf.keras.backend.set_value(self.model.optimizer.learning_rate,lr*0.7)
					tf.keras.backend.set_value(self.model.optimizer.momentum,mo*0.7)
					self.model.layers[0].set_weights(self.weightH)
					self.model.layers[1].set_weights(self.weightO)
					if(epoch==999):
						print('soft to the end wait')
						info2(self.weightH,self.weightO)
					##
					print('soft wait')
					##
				else:

					info2(self.weightH,self.weightO)
					##
					print('soft undesire')
					##
					self.model.stop_training = True 

			self.previous=self.new

###########################################################################
def info(undes,c0,c1,c0index,c1index,inte,lo,ep):
	global undesire_
	global class0_
	global class1_
	global class0_index
	global class1_index
	global integ
	global Loss_inte
	global epp
	#Recording information after weight tuning
	undesire_=undes #Record undesired attractor
	class0_=c0 #Record class 0 samples after weight tuning
	class1_=c1 #Record class 1 samples after weight tuning
	class0_index=c0index #Record row of class 0 samples after weight tuning
	class1_index=c1index #Record row of class 1 samples after weight tuning
	integ=inte #For judging success of integrating
	Loss_inte=lo #Record loss after weight tuning
	epp=ep #Record how many epochs have been run
##########################################################################
def info2(W1,W2):
	global Layer1W
	global Layer2W
	Layer1W=W1 #For recording information after softening
	Layer2W=W2
#------------------------------------------------------------------------
def weight_tuning(X,y,epoch_,model,LTS_ID):
	global X_call
	global Y_call
	global LTS_X_ID
	global epp
	X_call=X
	Y_call=y
	LTS_X_ID=LTS_ID
	epoch=epoch_
	epp=0
	tf.keras.backend.set_value(model.optimizer.learning_rate,0.05)
	tf.keras.backend.set_value(model.optimizer.momentum,0.05)
	model.fit(X,y,epochs=epoch,callbacks=[myCallback()],verbose=0)

	return model
#######################################################################
def softening(X,y,model,LTS_ID,Regg,H1Node):
	
	global X_call
	global Y_call
	global LTS_X_ID
	global Layer1W
	global Layer2W
	X_call=X
	Y_call=y
	LTS_X_ID=LTS_ID

	epoch=1000 #1000 epochs
	Newreg=Regg*1.2 #set up the regularization rate as 1.2 times of the original
	S1=model.layers[0].get_weights() #Save weight
	S2=model.layers[1].get_weights()

	##
	print('soft starts')
	#print(S1)
	##
	
	Newmodel=Mymodel(Newreg,H1Node)	
	Newmodel.layers[0].set_weights(S1)
	Newmodel.layers[1].set_weights(S2)
	##
	#print(Newmodel.layers[0].get_weights())
	##
	tf.keras.backend.set_value(Newmodel.optimizer.learning_rate,0.05)
	tf.keras.backend.set_value(Newmodel.optimizer.momentum,0.05)
	Newmodel.fit(X,y,epochs=epoch,callbacks=[soften()],verbose=0)	

	##
	print('soft over')
	#print(Layer1W)
	##
	#tf.keras.backend.clear_session()
	model.layers[0].set_weights(Layer1W)
	model.layers[1].set_weights(Layer2W)
	
	return model	
#------------------------------------------------------------------------
def LTS(Xtrain_nor,y,model,H1node,Reg):                               
	dataVolume = Xtrain_nor.shape[0]     #data amount 768
	global undesire_
	global class0_
	global class1_
	global class0_index
	global class1_index
	global integ
	global Loss_inte
	global epp
	global SaveweightH
	global SaveweightO
	
	H1Node=H1node
	Regg=Reg

	LTS_sample=2 #n
	while LTS_sample<=dataVolume:
		#undesire_=0
		#epp=0
		class0_.clear()
		class1_.clear()
		class0_index.clear()
		class1_index.clear()
		conditionL=0 #To judge condition L
		LTS_X_index=[] #LTS sample row in N
		LTS_Y=[] #LTS y
		Class0Pool=[] #Record y of the sample for class 0
		Class1Pool=[] #Record y of the sample for class 1
		Class0Pool_index=[] #Record row of the sample for class 0
		Class1Pool_index=[] #Record row of the sample for class 1
		savesoft1=0	#Save weight before softening
		savesoft2=0 #Save weight before softening
		class1pooldif=[] #For the use of class 1 pool of all samples
		class0pooldif=[] #For the use of class 0 pool of all samples
		LTS_dif=[]	#Record squared error of LTS sample 
		##
		print('LTS starts')
		print(LTS_sample)
		##
		#Predict N samples and sort
		predict=model.predict(Xtrain_nor)
		sq=np.square((predict-y))
		errMatrix = np.array(sq)     
		errMaSorted=np.sort(errMatrix,axis=0)          
		SortIndi = np.argsort(errMatrix,axis=0)
		
		#Set up N samples prediction into each pool
		for i in range(dataVolume):
			INDEX=SortIndi[i][0]
			your_turn=np.array(Xtrain_nor[INDEX,:]).reshape(1,-1)
			pp=model.predict(your_turn)
			ss=np.square((pp[0][0]-y[INDEX][0])).reshape(-1,1)
			if(pp[0][0]>0.5):
				Class1Pool.append(y[INDEX][0])		#put real y in the predictive class
				Class1Pool_index.append(INDEX)		#put index of respective x in
				class1pooldif.append(ss[0][0])
			else:
				Class0Pool.append(y[INDEX][0])
				Class0Pool_index.append(INDEX)
				class0pooldif.append(ss[0][0])
		
		count0=0
		count1=0
		class0_limit=len(Class0Pool)
		class1_limit=len(Class1Pool)
		#Extract LTS sample from each class
		for i in range(LTS_sample):
			if((i%2)==0):
				if(count0<class0_limit):
					class0_.append(Class0Pool[count0])
					class0_index.append(Class0Pool_index[count0])
					LTS_X_index.append(Class0Pool_index[count0])
					LTS_Y.append(Class0Pool[count0])
					LTS_dif.append(class0pooldif[count0])
					count0=count0+1
				else:
					class1_.append(Class1Pool[count1])
					class1_index.append(Class1Pool_index[count1])
					LTS_X_index.append(Class1Pool_index[count1])
					LTS_Y.append(Class1Pool[count1])
					LTS_dif.append(class1pooldif[count1])
					count1=count1+1
			if((i%2)==1):
				if(count1<class1_limit):
					class1_.append(Class1Pool[count1])
					class1_index.append(Class1Pool_index[count1])
					LTS_X_index.append(Class1Pool_index[count1])
					LTS_Y.append(Class1Pool[count1])
					LTS_dif.append(class1pooldif[count1])
					count1=count1+1
				else:
					class0_.append(Class0Pool[count0])
					class0_index.append(Class0Pool_index[count0])
					LTS_X_index.append(Class0Pool_index[count0])
					LTS_Y.append(Class0Pool[count0])
					LTS_dif.append(class0pooldif[count0])
					count0=count0+1
		#Sort extracted data based on squared error recorded
		LTS_X_index=sort_list(LTS_X_index,LTS_dif)
		LTS_Y=sort_list(LTS_Y,LTS_dif)

		#
		#LTS_dif=sort_list(LTS_dif,LTS_dif)
		#print(LTS_dif)
		#
		#Set up LTS sample
		LTS_X=np.take(a=Xtrain_nor,indices=LTS_X_index,axis=0)
		LTS_YY=np.array(LTS_Y).reshape(-1,1)	
		LTS_ID=np.array(LTS_X_index).reshape(-1,1)
		
		#Judge condition L
		alfa=np.amin(np.array(class1_)) 
		beta=np.amax(np.array(class0_))
		if(alfa>beta):
			conditionL=1
		else:
			conditionL=0

		##
		print(class0_)
		print(class1_)
		##
		
		if(conditionL==1):          
			SaveweightH=model.layers[0].get_weights()
			SaveweightO=model.layers[1].get_weights()
			model.layers[0].set_weights(SaveweightH)
			model.layers[1].set_weights(SaveweightO)
			#Integrating
			model,H1Node=integrating(model,H1Node,Regg,LTS_X,LTS_YY,LTS_ID)
			LTS_sample+=1
		
		else:
			#500 epochs
			epoch_=500
			#Save weight
			SaveweightH=model.layers[0].get_weights()		
			SaveweightO=model.layers[1].get_weights()	
			print('start tune')
			#weight tuning
			model=weight_tuning(LTS_X,LTS_YY,epoch_,model,LTS_ID)
			print('end tune')
			integ=0
			Loss_inte=0

			##
			#if(undesire_==0):
				#print('success')
				#print(undesire_)
				#print(epp)
				#print(class0_)
				#print(class1_)
			
			##	

			if((undesire_==1) or (epp==(epoch_))):
				#Set up weight before weight tuning
				model.layers[0].set_weights(SaveweightH)
				model.layers[1].set_weights(SaveweightO)
				#Get notfamiliar data
				NFindex=LTS_ID[LTS_sample-1][0]
				NotFamiliar_X = np.array(Xtrain_nor[NFindex,:]).reshape(1,-1)	#last element of LTS sample x
				NotFamiliar_Y=y[NFindex][0]	#last element of LTS sample real y
				NotFamiliar_predict=model.predict(NotFamiliar_X) #predict value of last element of LTS
				#Xc-Xn
				LTS_X_index.remove(NFindex)
				
				X_ex=np.take(a=Xtrain_nor,indices=LTS_X_index,axis=0)

				if(NotFamiliar_Y==1):
					boundary=beta
					X_minus_xn=X_ex-NotFamiliar_X
					

				if(NotFamiliar_Y==0):
					boundary=alfa
					X_minus_xn=X_ex-NotFamiliar_X
				
				##
				#print('weight before cram')
				#print(model.layers[0].get_weights())
				#before_cram_pre=model.predict(LTS_X).reshape(1,-1)
				#print(before_cram_pre)
				##
				#Cramming
				H1Node=H1Node+3
				model=cramming(X_minus_xn,boundary,NotFamiliar_predict,NotFamiliar_X,model,H1Node,Regg)

				##
				#after_cram_pre=model.predict(LTS_X).reshape(1,-1)
				#print(after_cram_pre)
				#print('weight after cram')
				#print(model.layers[0].get_weights())
				##
				LTS_X_index.append(NFindex)
				
				class0_.clear()
				class1_.clear()
				class0_index.clear()
				class1_index.clear()
				for i in range(len(LTS_X_index)):
					again=np.array(LTS_X[i,:]).reshape(1,-1)
					agp=model.predict(again)
					if(agp[0][0]>0.5):
						class1_.append(LTS_YY[i][0])		
						class1_index.append(LTS_X_index[i])	
					else:
						class0_.append(LTS_YY[i][0])		
						class0_index.append(LTS_X_index[i])	
				##
				print('after cram')
				print(class0_)
				print(class1_)
				##
				#integrating
				model,H1Node=integrating(model,H1Node,Regg,LTS_X,LTS_YY,LTS_ID)
				
			else:
				#integrating
				model,H1Node=integrating(model,H1Node,Regg,LTS_X,LTS_YY,LTS_ID)	
			
			LTS_sample+=1
			

	return model
			
#--------------------------------------------------------------------------  

def cramming(X_minus_xn,boundary,NotFamiliar_predict,NotFamiliar_X,model,H1Node,Regg):
	caci=1e-4
	result=0
	#Find gamma
	while result>=0:
		#r=np.random.rand(1,8)
		r=np.random.uniform(-1,1,(1,8))
		r2=np.square(r)
		s=r2.sum()
		sq=math.sqrt(s)	#To make it length as 1
		gamma=np.divide(r,sq)
		mm=np.matmul(X_minus_xn,np.transpose(gamma))
		plus=caci+mm
		minus=caci-mm
		result=np.matmul(np.transpose(plus),minus)
		result=result[0][0]
	#New 3 hidden layer bias
	WHo_2=caci-np.matmul(NotFamiliar_X,np.transpose(gamma))
	WHo_1=-np.matmul(NotFamiliar_X,np.transpose(gamma))
	WHo_0=-caci-np.matmul(NotFamiliar_X,np.transpose(gamma))
	
	#New 3 hidden layer weight
	WH_2=np.transpose(gamma)
	WH_1=np.transpose(gamma)
	WH_0=np.transpose(gamma)
	#New 3 output layer weight
	WO_0=(1.1)*(boundary-NotFamiliar_predict)/caci  
	WO_1=(-2)*WO_0
	WO_2=WO_0
	#Original weight
	OldweightH=model.layers[0].get_weights()
	OldweightO=model.layers[1].get_weights()

	#Append the weight of Hidden Layer
	New_Hidden_O=np.append(OldweightH[1],WHo_2)
	New_Hidden_O=np.append(New_Hidden_O,WHo_1)
	New_Hidden_O=np.append(New_Hidden_O,WHo_0)

	New_Hidden_W=np.concatenate((OldweightH[0],WH_2,WH_1,WH_0),axis=1).reshape(8,-1)
	#Append the weight of Output Layer
	New_Output_W=np.append(OldweightO[0],WO_2)
	New_Output_W=np.append(New_Output_W,WO_1)
	New_Output_W=np.append(New_Output_W,WO_0)
	New_Output_W=New_Output_W.reshape(-1,1)

	New_Output_O=OldweightO[1]
	#
	#Set up the weight of new model
	tf.keras.backend.clear_session()

	Newmodel=Mymodel(Regg,H1Node)
	L=[]
	L.append(New_Hidden_W)
	L.append(New_Hidden_O)
	Newmodel.layers[0].set_weights(L)

	K.set_value(Newmodel.layers[1].weights[0], New_Output_W)
	K.set_value(Newmodel.layers[1].weights[1], New_Output_O)
	
	return Newmodel
############################################################################
def integrating(model,H1Node,Regg,X,Y,ID):
	global integ
	global Loss_inte
	global savesoft1
	global savesoft2
	global SaveweightH
	global SaveweightO
	temploss=0 #record loss of reduced weight model
	tempnode=0 #reduced hidden nodes
	first=0	   #judge whether hidden node has been reduced
	

	if(H1Node>1):
		#Initial softining
		savesoft1=model.layers[0].get_weights()
		savesoft2=model.layers[1].get_weights()
		print('ini soft')
		model=softening(X,Y,model,ID,Regg,H1Node)
		print('ini soft end')
		#Save weight
		weightL1=model.layers[0].get_weights()
		weightL2=model.layers[1].get_weights()
		weightHW=weightL1[0]
		weightHO=weightL1[1]
		weightOW=weightL2[0]
		weightOO=weightL2[1]
		#500 epoch
		epoch_=500
		#Set up testing model of reduced hidden nodes
		#Attempt to reduce only 1 hidden
		Testmodel=Mymodel(Regg,H1Node-1)
		LL=[i for i in range(H1Node)]
		#Combinations of reduced hidden nodes
		comb = combinations(LL, (H1Node-1))
		COMB=list(comb)
		print(COMB)

		for j in range(len(COMB)):
			integ=0
			Loss_inte=0
			SEL=COMB[j]
			#Extract the reduced hidden nodes
			HWsel=np.take(a=weightHW,indices=SEL,axis=1)
			HOsel=np.take(a=weightHO,indices=SEL,axis=0)
			OWsel=np.take(a=weightOW,indices=SEL,axis=0)
			OOsel=weightOO
			#Set up testing model of reduced hidden nodes
			T=[]
			T.append(HWsel)
			T.append(HOsel)
			Testmodel.layers[0].set_weights(T)

			K.set_value(Testmodel.layers[1].weights[0], OWsel)
			K.set_value(Testmodel.layers[1].weights[1], OOsel)

			print('integrating start')
			#print(SEL)
			SaveweightH=Testmodel.layers[0].get_weights()		
			SaveweightO=Testmodel.layers[1].get_weights()
			#Weight tuning the model of reduced hidden nodes
			Temp=weight_tuning(X,Y,epoch_,Testmodel,ID)
			if(integ==1):
				if(first==0):
					#success of reducing hidden node
					first=first+1
					#store its loss
					temploss=Loss_inte
					#store the model of reduced hidden nodes
					finalModel=Temp
					tempnode=H1Node-1
					print('Hidden node reduced first')
					#print('weight before integrating')
					#print(model.layers[0].get_weights())
					#print('weight after integrating')
					#print(finalModel.layers[0].get_weights())
					print('ss start')
					#softening
					savesoft1=finalModel.layers[0].get_weights()
					savesoft2=finalModel.layers[1].get_weights()
					finalModel=softening(X,Y,finalModel,ID,Regg,tempnode)
					print('ss end')
				else:
					if(Loss_inte<temploss):
						#better reduced model
						temploss=Loss_inte
						finalModel=Temp
						tempnode=H1Node-1
						print('Hidden node reduced more')
						print('ss start')
						#softening
						savesoft1=finalModel.layers[0].get_weights()
						savesoft2=finalModel.layers[1].get_weights()
						finalModel=softening(X,Y,finalModel,ID,Regg,tempnode)
						print('ss end')
			else:
				if(first==0):
					finalModel=model
					tempnode=H1Node
					print('still no')
					#print('ss start')
					#savesoft1=finalModel.layers[0].get_weights()
					#savesoft2=finalModel.layers[1].get_weights()
					#finalModel=softening(X,Y,finalModel,ID,Regg,tempnode)
					#print('ss end')
				else:
					print('already reduced')
			
	else:
		finalModel=model
		tempnode=H1Node

	
	integ=0
	Loss_inte=0

	#tf.keras.backend.clear_session()
	return finalModel,tempnode
#############################################################################
#Initial hidden node
Ini_H1node=1
#Regularization rate
Ini_Reg=0.00001
#Data
df = pd.read_csv(r'C:\Users\chrystal212\Desktop\AI_project\diabetes.csv')  
#Eight feature
X_train=np.array(df.iloc[:,0:8])
Y_train=np.array(df.iloc[:,8]).reshape(-1,1)
#Data Preprocessing
scaler=MinMaxScaler(feature_range=(0,1))
Xtrain_nor=scaler.fit_transform(X_train,Y_train)
#Set up initial model
model=Mymodel(Ini_Reg,Ini_H1node)

######################################################################
#Set up initial weight
'''
x_test=np.array(Xtrain_nor[0:2,:])
y_test=np.array(Y_train[0:2,:])

condition1=0
condition2=0
regulizer=keras.regularizers.l2(l=0.00001)
optimizer=keras.optimizers.SGD(learning_rate=0.05,momentum=0.05,nesterov=True)
inimodel=keras.Sequential()
inimodel.add(tf.keras.layers.Dense(units=1,input_dim=8, name="hiddenLayer",activation='relu',use_bias=True,dynamic=True,kernel_regularizer=regulizer,bias_regularizer=regulizer))	
inimodel.add(tf.keras.layers.Dense(units=1, name="outputLayer",activation='linear',use_bias=True,dynamic=True,kernel_regularizer=regulizer,bias_regularizer=regulizer))
inimodel.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

while((condition1*condition2)==0):
	r1=np.random.rand(8,1)
	r2=np.random.rand(1)
	r3=np.random.rand(1,1)
	r4=np.random.rand(1)
	K.set_value(inimodel.layers[0].weights[0], r1)
	K.set_value(inimodel.layers[0].weights[1], r2)
	K.set_value(inimodel.layers[1].weights[0], r3)
	K.set_value(inimodel.layers[1].weights[1], r4)
	predict=inimodel.predict(x_test)
	real1=y_test[0]
	predict1=predict[0][0]
	if(predict1>0.5):
		result1=1
	else:
		result1=0

	if(real1[0]==result1):
		condition1=1
	else:
		condition1=0

	real2=y_test[1]
	predict2=predict[1][0]
	if(predict2>0.5):
		result2=1
	else:
		result2=0

	if(real2[0]==result2):
		condition2=1
	else:
		condition2=0
'''
#########################################################
#Transfer learning
r1=[[0.6551119],[0.09545052],[0.02110529],[0.34002292],[0.99083555],[0.33055994],[0.82845074],[0.49274734]]
r2=[0.12430776]
r1=np.array(r1).reshape(8,1)
r2=np.array(r2).reshape(1,)
r3=[[0.51014334]]
r4=[0.13120504]
r3=np.array(r3).reshape(1,1)
r4=np.array(r4).reshape(1,)

K.set_value(model.layers[0].weights[0], r1)
K.set_value(model.layers[0].weights[1], r2)
K.set_value(model.layers[1].weights[0], r3)
K.set_value(model.layers[1].weights[1], r4)

#Training 300 dataset
x_300=np.array(Xtrain_nor[0:300,:])
y_300=np.array(Y_train[0:300,:])
model=LTS(x_300,y_300,model,Ini_H1node,Ini_Reg)
#

#Testing 300~768 dataset
x_test=np.array(Xtrain_nor[300:,:]).reshape(-1,8)
y_test=np.array(Y_train[300:,:]).reshape(-1,1)
p_test=model.predict(x_test).reshape(-1,1)
#Print the prediction rate
p_test_trans=np.where(p_test>0.5,1,0)
p_test_trans=p_test_trans.reshape(-1,1)
diff=np.absolute((p_test_trans-y_test)).reshape(-1,1)
non_zero=np.count_nonzero(diff)
precision=(diff.shape[0]-non_zero)*(1/diff.shape[0])
print('precision')
print(precision)









