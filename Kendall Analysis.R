#Sampling from the Kendall Shape Space, i.e The surface of the 3D upper hemisphere.
#For the acute/ obtuse exploration we generate random coordinates uniformly on the sphere. 
# We do this by generating coordinates using six independent standard normal distributions, converting these
# to Kendall spherical coordinates to use as inputs into the neural network.
# Because the space on the hemisphere is finite we have no need to restrict where we sample from. 

library(keras)
library(shapes)
library(latex2exp)
library(scatterplot3d)
library(rgl)
library(plot3D)
library(plot3Drgl)

data_ken <- function(N){
  phi <- runif(N,0,2*pi)
  theta <- acos(runif(N,0,1))
  u <- sin(theta)*sin(phi)/(1+sin(theta)*cos(phi))
  v <- cos(theta)/(1+sin(theta)*cos(phi))
  return(list(polar = cbind(theta,phi),plane = cbind(u,v)))
} #Generates Nx2 matrix of kendall polar coordinates that are uniformly
#distributed on the sphere.

kendall.ao <- function(X){
  acute_u <- c()
  acute_v <- c()
  obtuse_u <- c()
  obtuse_v <- c()
  index_acute <- c()
  index_obtuse <- c()
  
  for (i in 1:dim(X)[1]){
    if ( (-1/sqrt(3) < X[i,1]) & (X[i,1] < 1/sqrt(3)) & (X[i,1]**2 + X[i,2]**2 > 1/3) ){
      acute_u <- append(acute_u,X[i,1])
      acute_v <- append(acute_v,X[i,2])
      index_acute <- append(index_acute,i)
    }
    else{
      obtuse_u <- append(obtuse_u,X[i,1])
      obtuse_v <- append(obtuse_v,X[i,2])
      index_obtuse <- append(index_obtuse,i)
    }
  }
  
  acute_data <- cbind(matrix(acute_u),matrix(acute_v))
  obtuse_data <- cbind(matrix(obtuse_u),matrix(obtuse_v))
  
  return(list(acute = acute_data, obtuse = obtuse_data, index_a = index_acute, index_o = index_obtuse))
} #Sorts the data set into acute and obtuse triangles 

scatter3Drgl(0.5*sin(theta)*cos(phi),0.5*sin(theta)*sin(phi),0.5*cos(theta),aspect3d(x = 1,y = 1,z = 1))

#Investigation 1: Acute or Obtuse----

#NN_1_hidden_layer computes runs a 1 hidden layer neural network with 'k' hidden neurons  
# on N pieces of training data. We can also decide the batch size and number of epochs to use
NN_1_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels 
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}

result1_100 <- NN_1_hidden_layer(100,k = 100,epoch = 10,batch = 10)
result1_500 <- NN_1_hidden_layer(500,k = 100,epoch = 10,batch = 10)
result1_1000 <- NN_1_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result1_10000 <- NN_1_hidden_layer(10000,k = 100,epoch = 10,batch = 10)

#Neural Network 2 - multiple hidden layers
NN_multi_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels 
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
    layer_dense(units = k,activation = 'relu') %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}

result_multi_100 <- NN_multi_hidden_layer(100,k = 4,epoch = 10,batch = 10)
result_multi_500 <- NN_multi_hidden_layer(500,k = 4,epoch = 10,batch = 10)
result_multi_1000 <- NN_multi_hidden_layer(1000,k = 4,epoch = 10,batch = 10)
result_multi_10000 <- NN_multi_hidden_layer(10000,k = 4,epoch = 10,batch = 10)

#Neural Network 3 - 1 hidden layer, different activation functions
NN_lReLu_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels  
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,input_shape = c(2)) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}
NN_pReLu_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels 
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,input_shape = c(2)) %>%
    layer_activation_parametric_relu() %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}
NN_seLu_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels 
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,input_shape = c(2)) %>%
    layer_activation_selu() %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}
NN_tanh_hidden_layer <- function(N,k,epoch,batch){
  train.kendall <- data_ken(N) #Initial data in kendall plane coordinates
  test.kendall <- data_ken(5000) 
  
  ao.traindata <- rbind(train.kendall$polar[kendall.ao(train.kendall$plane)$index_a,],
                        train.kendall$polar[kendall.ao(train.kendall$plane)$index_o,]) 
  ao.testdata <- rbind(test.kendall$polar[kendall.ao(test.kendall$plane)$index_a,],
                       test.kendall$polar[kendall.ao(test.kendall$plane)$index_o,]) 
  #Training data now in correct format of polar coordinates 
  #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(kendall.ao(train.kendall$plane)$acute)[1])),
                          matrix(rep(0,dim(kendall.ao(train.kendall$plane)$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(kendall.ao(test.kendall$plane)$acute)[1])),
                         matrix(rep(0,dim(kendall.ao(test.kendall$plane)$obtuse)[1]))) #Test labels 
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'tanh',input_shape = c(2)) %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(ao.traindata,to_categorical(ao.trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(ao.testdata,to_categorical(ao.testlabels))
  pred <- network %>% predict(ao.testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(ao.testdata)[1],1)
  
  correctpred <- ao.testdata[which(pred == ao.testlabels),]
  incorrectpred <- ao.testdata[-which(pred == ao.testlabels),]
  
  return(metrics)
}

result_lReLu <- NN_lReLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_pReLu <- NN_pReLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_seLu <- NN_seLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_tanh <- NN_tanh_hidden_layer(1000,k = 100,epoch = 10,batch = 10)




