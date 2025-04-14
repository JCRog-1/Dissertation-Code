library(keras)
library(shapes)
library(latex2exp)
library(tensorflow)

#This file should be run in tandem with large_dataset.R, there are functions required to be in the global environment in order for this file to run
#Investigation 1: Acute or Obtuse ----
dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA AO", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_rawda_1_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  
  data_aug <- t(apply(samp$triangles,MARGIN = 3,FUN = function(x) rand_affine(t(x))))
  
  traindata <- rbind(t(array_reshape(samp$triangles, dim = c(6,N),order = 'F')),data_aug)
  testdata <- t(array_reshape(triangle_data_test$triangles, dim = c(6,dim(triangle_data_test$triangles)[3]),order = 'F'))
  
  trainlabels <- rbind(samp$ao,samp$ao)
  testlabels <- triangle_data_test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(6)) %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)

  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA AO/ao-rdabk-single-",N,'-',k,'-notrain.pdf'))
  #Comparative plot on the Hyperbolic Plane
  
  bkcorrectpred <- bookstein.test$data[which(pred == testlabels),]
  bkincorrectpred <- bookstein.test$data[-which(pred == testlabels),]
  
  plot(bkcorrectpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Bookstein, Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(bkincorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA AO/ao-rdakd-single-",N,'-',k,'-notrain.pdf'))
  
  #Comparative plot on the Kendall Sphere
  kdcorrectpred <- kendall.test$cart[which(pred == testlabels),]
  kdincorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  X_correct <- kdcorrectpred[,1]/(1-kdcorrectpred[,3])
  Y_correct <- kdcorrectpred[,2]/(1-kdcorrectpred[,3])
  X_incorrect <- kdincorrectpred[,1]/(1-kdincorrectpred[,3])
  Y_incorrect <- kdincorrectpred[,2]/(1-kdincorrectpred[,3])
  
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Kendall, Training size = ',N,', Network = Single Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  lines(X_border[1:1000],Y_border[1:1000],col = 'black',lwd = 2)
  lines(X_border[1001:2000],Y_border[1001:2000],col = 'black',lwd = 2)
  lines(X_border[2001:3000],Y_border[2001:3000],col = 'black',lwd = 2)
  lines(X_border[3001:4000],Y_border[3001:4000],col = 'black',lwd = 2)
  lines(X_border[4001:5000],Y_border[4001:5000],col = 'black',lwd = 2)
  lines(X_border[5001:6000],Y_border[5001:6000],col = 'black',lwd = 2)
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_ao(which(pred != testlabels))
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.rawda.N10.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.rawda.N50.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.rawda.N100.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.rawda.N500.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.rawda.N1000.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.rawda.N10000.k4 <- NN_rawda_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.rawda.N10.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.rawda.N50.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.rawda.N100.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.rawda.N500.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.rawda.N1000.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.rawda.N10000.k100 <- NN_rawda_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

rawda_1_hidden_layer_ao <- list(result.rawda.N10.k4, result.rawda.N50.k4,result.rawda.N100.k4,result.rawda.N500.k4,result.rawda.N1000.k4,result.rawda.N10000.k4,
                              result.rawda.N10.k100, result.rawda.N50.k100,result.rawda.N100.k100,result.rawda.N500.k100,result.rawda.N1000.k100,result.rawda.N10000.k100)
saveRDS(rawda_1_hidden_layer_ao,file = 'rawda_1_hidden_layer_ao')

#Multi Hidden layer 
#A function that runs a two layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_rawda_multi_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  
  data_aug <- t(apply(samp$triangles,MARGIN = 3,FUN = function(x) rand_affine(t(x))))
  
  traindata <- rbind(t(array_reshape(samp$triangles, dim = c(6,N),order = 'F')),data_aug)
  testdata <- t(array_reshape(triangle_data_test$triangles, dim = c(6,dim(triangle_data_test$triangles)[3]),order = 'F'))
  
  trainlabels <- rbind(samp$ao,samp$ao)
  testlabels <- triangle_data_test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(6)) %>%
    layer_dense(units = k,activation = 'relu') %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)

  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA AO/ao-rdabk-multi-",N,'-',k,'-notrain.pdf'))
  #Comparative plot on the Hyperbolic Plane
  
  bkcorrectpred <- bookstein.test$data[which(pred == testlabels),]
  bkincorrectpred <- bookstein.test$data[-which(pred == testlabels),]
  
  plot(bkcorrectpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Bookstein, Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(bkincorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA AO/ao-rdakd-multi-",N,'-',k,'-notrain.pdf'))
  
  #Comparative plot on the Kendall Sphere
  kdcorrectpred <- kendall.test$cart[which(pred == testlabels),]
  kdincorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  X_correct <- kdcorrectpred[,1]/(1-kdcorrectpred[,3])
  Y_correct <- kdcorrectpred[,2]/(1-kdcorrectpred[,3])
  X_incorrect <- kdincorrectpred[,1]/(1-kdincorrectpred[,3])
  Y_incorrect <- kdincorrectpred[,2]/(1-kdincorrectpred[,3])
  
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Kendall, Training size = ',N,', Network = Multi Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  lines(X_border[1:1000],Y_border[1:1000],col = 'black',lwd = 2)
  lines(X_border[1001:2000],Y_border[1001:2000],col = 'black',lwd = 2)
  lines(X_border[2001:3000],Y_border[2001:3000],col = 'black',lwd = 2)
  lines(X_border[3001:4000],Y_border[3001:4000],col = 'black',lwd = 2)
  lines(X_border[4001:5000],Y_border[4001:5000],col = 'black',lwd = 2)
  lines(X_border[5001:6000],Y_border[5001:6000],col = 'black',lwd = 2)
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_ao(which(pred != testlabels))
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.rawda_multi.N10.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.rawda_multi.N50.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.rawda_multi.N100.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.rawda_multi.N500.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.rawda_multi.N1000.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.rawda_multi.N10000.k4 <- NN_rawda_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.rawda_multi.N10.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.rawda_multi.N50.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.rawda_multi.N100.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.rawda_multi.N500.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.rawda_multi.N1000.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.rawda_multi.N10000.k100 <- NN_rawda_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

rawda_multi_hidden_layer_ao <- list(result.rawda_multi.N10.k4, result.rawda_multi.N50.k4,result.rawda_multi.N100.k4,result.rawda_multi.N500.k4,result.rawda_multi.N1000.k4,result.rawda_multi.N10000.k4,
                                  result.rawda_multi.N10.k100, result.rawda_multi.N50.k100,result.rawda_multi.N100.k100,result.rawda_multi.N500.k100,result.rawda_multi.N1000.k100,result.rawda_multi.N10000.k100)
saveRDS(rawda_multi_hidden_layer_ao,file = 'rawda_multi_hidden_layer_ao')

#Investigation 2 - Shape -----

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA Shape", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_rawda_1_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  
  data_aug <- t(apply(samp$triangles,MARGIN = 3,FUN = function(x) rand_affine(t(x))))
  
  traindata <- rbind(t(array_reshape(samp$triangles, dim = c(6,N),order = 'F')),data_aug)
  testdata <- t(array_reshape(triangle_data_test$triangles, dim = c(6,dim(triangle_data_test$triangles)[3]),order = 'F'))
  
  trainlabels <- rbind(samp$shape,samp$shape)
  testlabels <- triangle_data_test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(6)) %>%
    layer_dense(units = 3,activation = 'softmax')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)

  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA Shape/shape-rdabk-single-",N,'-',k,'-notrain.pdf'))
  #Comparative plot on the Hyperbolic Plane
  
  bkcorrectpred <- bookstein.test$data[which(pred == testlabels),]
  bkincorrectpred <- bookstein.test$data[-which(pred == testlabels),]
  
  plot(bkcorrectpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(bkincorrectpred,pch = 20, col = 'red')
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA Shape/shape-rdakd-single-",N,'-',k,'-notrain.pdf'))
  
  #Comparative plot on the Kendall Sphere
  kdcorrectpred <- kendall.test$cart[which(pred == testlabels),]
  kdincorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  X_correct <- kdcorrectpred[,1]/(1-kdcorrectpred[,3])
  Y_correct <- kdcorrectpred[,2]/(1-kdcorrectpred[,3])
  X_incorrect <- kdincorrectpred[,1]/(1-kdincorrectpred[,3])
  Y_incorrect <- kdincorrectpred[,2]/(1-kdincorrectpred[,3])
  
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Single Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_shape(which(pred != testlabels))
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.rawda.M10.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.rawda.M50.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.rawda.M100.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.rawda.M500.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.rawda.M1000.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.rawda.M10000.k4 <- NN_rawda_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.rawda.M10.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.rawda.M50.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.rawda.M100.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.rawda.M500.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.rawda.M1000.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.rawda.M10000.k100 <- NN_rawda_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

rawda_1_hidden_layer_shape <- list(result.rawda.M10.k4, result.rawda.M50.k4,result.rawda.M100.k4,result.rawda.M500.k4,result.rawda.M1000.k4,result.rawda.M10000.k4,
                                 result.rawda.M10.k100, result.rawda.M50.k100,result.rawda.M100.k100,result.rawda.M500.k100,result.rawda.M1000.k100,result.rawda.M10000.k100)
saveRDS(rawda_1_hidden_layer_shape,file = 'rawda_1_hidden_layer_shape')

#Multi Hidden layer 
#A function that runs a two layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_rawda_multi_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  
  data_aug <- t(apply(samp$triangles,MARGIN = 3,FUN = function(x) rand_affine(t(x))))
  
  traindata <- rbind(t(array_reshape(samp$triangles, dim = c(6,N),order = 'F')),data_aug)
  testdata <- t(array_reshape(triangle_data_test$triangles, dim = c(6,dim(triangle_data_test$triangles)[3]),order = 'F'))
  
  trainlabels <- rbind(samp$shape,samp$shape)
  testlabels <- triangle_data_test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(6)) %>%
    layer_dense(units = k,activation = 'relu') %>%
    layer_dense(units = 3,activation = 'softmax')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)

  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA Shape/shape-rdabk-multi-",N,'-',k,'-notrain.pdf'))
  #Comparative plot on the Hyperbolic Plane
  
  bkcorrectpred <- bookstein.test$data[which(pred == testlabels),]
  bkincorrectpred <- bookstein.test$data[-which(pred == testlabels),]
  
  plot(bkcorrectpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(bkincorrectpred,pch = 20, col = 'red')
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/RawDA Shape/shape-rdakd-multi-",N,'-',k,'-notrain.pdf'))
  
  #Comparative plot on the Kendall Sphere
  kdcorrectpred <- kendall.test$cart[which(pred == testlabels),]
  kdincorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  X_correct <- kdcorrectpred[,1]/(1-kdcorrectpred[,3])
  Y_correct <- kdcorrectpred[,2]/(1-kdcorrectpred[,3])
  X_incorrect <- kdincorrectpred[,1]/(1-kdincorrectpred[,3])
  Y_incorrect <- kdincorrectpred[,2]/(1-kdincorrectpred[,3])
  
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Multi Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_shape(which(pred != testlabels))
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.rawda_multi.M10.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.rawda_multi.M50.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.rawda_multi.M100.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.rawda_multi.M500.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.rawda_multi.M1000.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.rawda_multi.M10000.k4 <- NN_rawda_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.rawda_multi.M10.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.rawda_multi.M50.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.rawda_multi.M100.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.rawda_multi.M500.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.rawda_multi.M1000.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.rawda_multi.M10000.k100 <- NN_rawda_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

rawda_multi_hidden_layer_shape <- list(result.rawda_multi.M10.k4, result.rawda_multi.M50.k4,result.rawda_multi.M100.k4,result.rawda_multi.M500.k4,result.rawda_multi.M1000.k4,result.rawda_multi.M10000.k4,
                                     result.rawda_multi.M10.k100, result.rawda_multi.M50.k100,result.rawda_multi.M100.k100,result.rawda_multi.M500.k100,result.rawda_multi.M1000.k100,result.rawda_multi.M10000.k100)
saveRDS(rawda_multi_hidden_layer_shape,file = 'rawda_multi_hidden_layer_shape')


