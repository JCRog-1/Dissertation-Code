#Sampling from the Bookstein Shape Space, i.e The hyperbolic plane.
#For the acute/ obtuse exploration we generate random coordinates from a uniform 
# distribution across the plane from the box [-z - 1.5,1.5 + z]x(0,1 + z] where z will be defined
# below in the code. This is so that the area inside the half circles is equal to the area outside
# the half circles. 

library(keras)
library(shapes)
library(latex2exp)

data <- function(N,height){
  y <- 1/2 - pi/(8*height)
  ub <- runif(N,min = -0.5 - y, max = 0.5 + y)
  vb <- runif(N,min = 0, max = height)
  
  return(cbind(ub,vb))
} #Function that generates the data

#Investigation 1: Acute or Obtuse ----
bookstein.ao <- function(X){
  acute_u <- c()
  acute_v <- c()
  obtuse_u <- c()
  obtuse_v <- c()
  
  for (i in 1:dim(X)[1]){
    if ( (-0.5 < X[i,1]) & (X[i,1] < 0.5) & (X[i,1]**2 + X[i,2]**2 > 0.25) ){
      acute_u <- append(acute_u,X[i,1])
      acute_v <- append(acute_v,X[i,2])
    }
    else{
      obtuse_u <- append(obtuse_u,X[i,1])
      obtuse_v <- append(obtuse_v,X[i,2])
    }
  }
  
  acute_data <- cbind(matrix(acute_u),matrix(acute_v))
  obtuse_data <- cbind(matrix(obtuse_u),matrix(obtuse_v))
  
  return(list(acute = acute_data, obtuse = obtuse_data))
} #Sorts the data set into acute and obtuse triangles 

#Neural Network 1 - 1 hidden layer - 100 pieces of data

#NN_1_hidden_layer computes runs a 1 hidden layer neural network with 'k' hidden neurons  
# on N pieces of training data. We can also decide the batch size and number of epochs to use
NN_1_hidden_layer <- function(N,k,epoch,batch,height){
  train.bookstein <- data(N,height) 
  test.bookstein <- data(5000,height) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  y <- 1/2 - pi/(8*height)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(height,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}

result1_100 <- NN_1_hidden_layer(100,k = 100,epoch = 10,batch = 10,10)
result1_500 <- NN_1_hidden_layer(500,k = 100,epoch = 10,batch = 10)
result1_1000 <- NN_1_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result1_10000 <- NN_1_hidden_layer(10000,k = 100,epoch = 10,batch = 10)

#Neural Network 2 - multiple hidden layers
NN_multi_hidden_layer <- function(N,k,epoch,batch){
  train.bookstein <- data(N) 
  test.bookstein <- data(5000) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  x <- 1.5
  y <- 1/2 - pi/(8*x)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(1.5,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}

result_multi_100 <- NN_multi_hidden_layer(100,k = 4,epoch = 10,batch = 10)
result_multi_500 <- NN_multi_hidden_layer(500,k = 4,epoch = 10,batch = 10)
result_multi_1000 <- NN_multi_hidden_layer(1000,k = 4,epoch = 10,batch = 10)
result_multi_10000 <- NN_multi_hidden_layer(10000,k = 4,epoch = 10,batch = 10)

#Neural Network 3 - 1 hidden layer, different activation functions
NN_lReLu_hidden_layer <- function(N,k,epoch,batch){
  train.bookstein <- data(N) 
  test.bookstein <- data(5000) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  x <- 1.5
  y <- 1/2 - pi/(8*x)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(1.5,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}
NN_pReLu_hidden_layer <- function(N,k,epoch,batch){
  train.bookstein <- data(N) 
  test.bookstein <- data(5000) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  x <- 1.5
  y <- 1/2 - pi/(8*x)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(1.5,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}
NN_seLu_hidden_layer <- function(N,k,epoch,batch){
  train.bookstein <- data(N) 
  test.bookstein <- data(5000) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  x <- 1.5
  y <- 1/2 - pi/(8*x)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(1.5,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}
NN_tanh_hidden_layer <- function(N,k,epoch,batch){
  train.bookstein <- data(N) 
  test.bookstein <- data(5000) 
  
  ao.train <- bookstein.ao(train.bookstein) #Initial data
  ao.test <- bookstein.ao(test.bookstein)
  
  ao.traindata <- rbind(ao.train$acute,ao.train$obtuse) #Training data in correct input format
  ao.testdata <- rbind(ao.test$acute,ao.test$obtuse) #Test data in correct input format
  
  ao.trainlabels <- rbind(matrix(rep(1,dim(ao.train$acute)[1])),
                          matrix(rep(0,dim(ao.train$obtuse)[1])))#Training labels 1 or 0
  ao.testlabels <- rbind(matrix(rep(1,dim(ao.test$acute)[1])),
                         matrix(rep(0,dim(ao.test$obtuse)[1]))) #Test labels 
  
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
  
  x <- 1.5
  y <- 1/2 - pi/(8*x)
  plot(correctpred,asp = 1,pch = 20,col = 'dark blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  abline(v = 0.5 + y,lwd = 2)
  abline(v = -0.5-y,lwd = 2)
  topx <- seq(-0.5-y,0.5+y,length.out = 100)
  topy <- rep(1.5,100)
  lines(topx,topy,lwd = 2)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  return(metrics)
}

result_lReLu <- NN_lReLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_pReLu <- NN_pReLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_seLu <- NN_seLu_hidden_layer(1000,k = 100,epoch = 10,batch = 10)
result_tanh <- NN_tanh_hidden_layer(1000,k = 100,epoch = 10,batch = 10)

#Investigation 2: Equilateral, Isosceles, Scalene ----







plot(bookstein,asp = 1,pch = 20)
abline(v = 0,col = 'red',lwd = 2)
abline(h = 1+z,col = 'red',lwd = 2)
xleftcirc <- seq(-1.5,0.5, length.out = 1000)
yleftcirc <- sqrt(1 - (xleftcirc + 0.5)**2)
lines(xleftcirc,yleftcirc,col = 'red',lwd = 2)
xrightcirc <- seq(-0.5,1.5, length.out = 1000)
yrightcirc <- sqrt(1 - (xrightcirc - 0.5)**2)
lines(xrightcirc,yrightcirc,col = 'red',lwd = 2)










