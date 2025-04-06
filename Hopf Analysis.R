library(keras)
library(shapes)
library(latex2exp)

#This file should be run in tandem with large_dataset.R

#Investigation 1: Acute or Obtuse ----
#Neural Network 1 - 1 hidden layer - 100 pieces of data
#NN_1_hidden_layer computes runs a 1 hidden layer neural network with 'k' hidden neurons  
# on N pieces of training data. We can also decide the batch size and number of epochs to use

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf AO", recursive = TRUE)

NN_Hopf_1_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  Hopf_input <- Hopf.conversion(samp)
  
  traindata <- t(array_reshape(Hopf_input$fibre, dim = c(4,N), order = 'F'))
  testdata <- t(array_reshape(Hopf.test$fibre, dim = c(4,length(Hopf.test$ao)), order = 'F'))
  
  trainlabels <- Hopf_input$ao
  testlabels <- Hopf.test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(4)) %>%
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

  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf AO/ao-Hbk-single-",N,'-',k,'-notrain.pdf'))
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf AO/ao-Hkd-single-",N,'-',k,'-notrain.pdf'))
  
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.Hopf.N10.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.Hopf.N50.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.Hopf.N100.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.Hopf.N500.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.Hopf.N1000.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.Hopf.N10000.k4 <- NN_Hopf_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.Hopf.N10.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.Hopf.N50.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.Hopf.N100.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.Hopf.N500.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.Hopf.N1000.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.Hopf.N10000.k100 <- NN_Hopf_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

Hopf_1_hidden_layer_ao <- list(result.Hopf.N10.k4, result.Hopf.N50.k4,result.Hopf.N100.k4,result.Hopf.N500.k4,result.Hopf.N1000.k4,result.Hopf.N10000.k4,
                             result.Hopf.N10.k100, result.Hopf.N50.k100,result.Hopf.N100.k100,result.Hopf.N500.k100,result.Hopf.N1000.k100,result.Hopf.N10000.k100)
saveRDS(Hopf_1_hidden_layer_ao,file = 'Hopf_1_hidden_layer_ao')

#Multi Hidden layer 

NN_Hopf_multi_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  Hopf_input <- Hopf.conversion(samp)
  
  traindata <- t(array_reshape(Hopf_input$fibre, dim = c(4,N), order = 'F'))
  testdata <- t(array_reshape(Hopf.test$fibre, dim = c(4,length(Hopf.test$ao)), order = 'F'))
  
  trainlabels <- Hopf_input$ao
  testlabels <- Hopf.test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(4)) %>%
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf AO/ao-Hbk-multi-",N,'-',k,'-notrain.pdf'))
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf AO/ao-Hkd-multi-",N,'-',k,'-notrain.pdf'))
  
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.Hopf_multi.N10.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.Hopf_multi.N50.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.Hopf_multi.N100.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.Hopf_multi.N500.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.Hopf_multi.N1000.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.Hopf_multi.N10000.k4 <- NN_Hopf_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.Hopf_multi.N10.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.Hopf_multi.N50.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.Hopf_multi.N100.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.Hopf_multi.N500.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.Hopf_multi.N1000.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.Hopf_multi.N10000.k100 <- NN_Hopf_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

Hopf_multi_hidden_layer_ao <- list(result.Hopf_multi.N10.k4, result.Hopf_multi.N50.k4,result.Hopf_multi.N100.k4,result.Hopf_multi.N500.k4,result.Hopf_multi.N1000.k4,result.Hopf_multi.N10000.k4,
                                 result.Hopf_multi.N10.k100, result.Hopf_multi.N50.k100,result.Hopf_multi.N100.k100,result.Hopf_multi.N500.k100,result.Hopf_multi.N1000.k100,result.Hopf_multi.N10000.k100)
saveRDS(Hopf_multi_hidden_layer_ao,file = 'Hopf_multi_hidden_layer_ao')

#Investigation 2 - Shape -----

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf Shape", recursive = TRUE)

NN_Hopf_1_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  Hopf_input <- Hopf.conversion(samp)
  
  traindata <- t(array_reshape(Hopf_input$fibre, dim = c(4,N), order = 'F'))
  testdata <- t(array_reshape(Hopf.test$fibre, dim = c(4,length(Hopf.test$ao)), order = 'F'))
  
  trainlabels <- Hopf_input$shape
  testlabels <- Hopf.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(4)) %>%
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf Shape/shape-Hbk-single-",N,'-',k,'-notrain.pdf'))
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf Shape/shape-Hkd-single-",N,'-',k,'-notrain.pdf'))
  
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.Hopf.M10.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.Hopf.M50.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.Hopf.M100.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.Hopf.M500.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.Hopf.M1000.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.Hopf.M10000.k4 <- NN_Hopf_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.Hopf.M10.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.Hopf.M50.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.Hopf.M100.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.Hopf.M500.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.Hopf.M1000.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.Hopf.M10000.k100 <- NN_Hopf_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

Hopf_1_hidden_layer_shape <- list(result.Hopf.M10.k4, result.Hopf.M50.k4,result.Hopf.M100.k4,result.Hopf.M500.k4,result.Hopf.M1000.k4,result.Hopf.M10000.k4,
                                result.Hopf.M10.k100, result.Hopf.M50.k100,result.Hopf.M100.k100,result.Hopf.M500.k100,result.Hopf.M1000.k100,result.Hopf.M10000.k100)
saveRDS(Hopf_1_hidden_layer_shape,file = 'Hopf_1_hidden_layer_shape')

#Multi Hidden layer 

NN_Hopf_multi_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  Hopf_input <- Hopf.conversion(samp)
  
  traindata <- t(array_reshape(Hopf_input$fibre, dim = c(4,N), order = 'F'))
  testdata <- t(array_reshape(Hopf.test$fibre, dim = c(4,length(Hopf.test$ao)), order = 'F'))
  
  trainlabels <- Hopf_input$shape
  testlabels <- Hopf.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(4)) %>%
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf Shape/shape-Hbk-multi-",N,'-',k,'-notrain.pdf'))
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
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Hopf Shape/shape-Hkd-multi-",N,'-',k,'-notrain.pdf'))
  
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.Hopf_multi.M10.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.Hopf_multi.M50.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.Hopf_multi.M100.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.Hopf_multi.M500.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.Hopf_multi.M1000.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.Hopf_multi.M10000.k4 <- NN_Hopf_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.Hopf_multi.M10.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.Hopf_multi.M50.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.Hopf_multi.M100.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.Hopf_multi.M500.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.Hopf_multi.M1000.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.Hopf_multi.M10000.k100 <- NN_Hopf_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000) 

Hopf_multi_hidden_layer_shape <- list(result.Hopf_multi.M10.k4, result.Hopf_multi.M50.k4,result.Hopf_multi.M100.k4,result.Hopf_multi.M500.k4,result.Hopf_multi.M1000.k4,result.Hopf_multi.M10000.k4,
                                    result.Hopf_multi.M10.k100, result.Hopf_multi.M50.k100,result.Hopf_multi.M100.k100,result.Hopf_multi.M500.k100,result.Hopf_multi.M1000.k100,result.Hopf_multi.M10000.k100)
saveRDS(Hopf_multi_hidden_layer_shape,file = 'Hopf_multi_hidden_layer_shape')

