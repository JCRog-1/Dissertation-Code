library(keras)
library(shapes)
library(latex2exp)
library(ggplot2)

#This file should be run in tandem with large_dataset.R

#Samples for AO investigation
N10 <- bk.conversion(n10)
N50 <- bk.conversion(n50)
N100 <- bk.conversion(n100)
N500 <- bk.conversion(n500)
N1000 <- bk.conversion(n1000)
N10000 <- bk.conversion(n10000)

#Samples for Shape investigation
M10 <- bk.conversion(m10)
M50 <- bk.conversion(m50)
M100 <- bk.conversion(m100)
M500 <- bk.conversion(m500)
M1000 <- bk.conversion(m1000)
M10000 <- bk.conversion(m10000)


#Investigation 1: Acute or Obtuse
dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO", recursive = TRUE)

#Neural Network 1 - 1 hidden layer ao ----
NN_bk_1_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  bookstein_input <- samp
  
  traindata <- bookstein_input$data
  testdata <- bookstein.test$data
    
  trainlabels <- bookstein_input$ao
  testlabels <- bookstein.test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[-which(pred == testlabels),]
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-single-large-notrain.pdf'))
  
  #Plot 1 
  plot(correctpred,asp = 1,pch = 20,col = 'blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k),las = 1)
  points(incorrectpred,pch = 20, col = 'red')
  abline(h = 0,lty = 3,lwd = 2.5)
  
  dev.off()
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-single-close-notrain.pdf'))
  #Plot 2
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-single-close-train.pdf'))

  #Plot 4
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  points(traindata,pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  distance.from.border <- apply(incorrectpred,MARGIN = 1, FUN = function(x) calc_dist_from_border_ao(x))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

#Results for 1 hidden layer different neurons ao ----
result.bk.N10.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N10,N = 10)
result.bk.N50.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N50,N = 50)
result.bk.N100.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N100,N = 100)
result.bk.N500.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N500,N = 500)
result.bk.N1000.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N1000,N = 1000)
result.bk.N10000.k4 <- NN_bk_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N10000,N = 10000)

result.bk.N10.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N10,N = 10)
result.bk.N50.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N50,N = 50)
result.bk.N100.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N100,N = 100)
result.bk.N500.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N500,N = 500)
result.bk.N1000.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N1000,N = 1000)
result.bk.N10000.k100 <- NN_bk_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N10000,N = 10000)

bk_1_hidden_layer_ao <- list(result.bk.N10.k4, result.bk.N50.k4,result.bk.N100.k4,result.bk.N500.k4,result.bk.N1000.k4,result.bk.N10000.k4,
                             result.bk.N10.k100, result.bk.N50.k100,result.bk.N100.k100,result.bk.N500.k100,result.bk.N1000.k100,result.bk.N10000.k100)
saveRDS(bk_1_hidden_layer_ao,file = 'bk_1_hidden_layer_ao')

#####
#Neural Network 2 - multiple hidden layers ao ----
NN_bk_multi__hidden_layer_ao <- function(k,epoch,batch,samp,N){
  bookstein_input <- samp
  
  traindata <- bookstein_input$data
  testdata <- bookstein.test$data
  
  trainlabels <- bookstein_input$ao
  testlabels <- bookstein.test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[-which(pred == testlabels),]
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-multi-large-notrain.pdf'))
  
  #Plot 1 
  plot(correctpred,asp = 1,pch = 20,col = 'blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,'Network = Multiple Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(h = 0,lty = 3,lwd = 2.5)
  
  dev.off()
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-multi-close-notrain.pdf'))
  
  #Plot 2
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,'Network = Multiple Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste0('/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein AO/ao-bk-',N,'-',k,'-multi-close-train.pdf'))
  
  #Plot 4
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,'Network = Multiple Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0.5,lty = 5,lwd = 2.5)
  abline(v = -0.5,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xracirc <- seq(-0.5,0.5, length.out = 1000)
  yracirc <- sqrt(0.25 - (xracirc)**2)
  lines(xracirc,yracirc,lty = 5,lwd = 3)
  points(traindata,pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  distance.from.border <- apply(incorrectpred,MARGIN = 1, FUN = function(x) calc_dist_from_border_ao(x))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}
#Results for multi hidden layers different neurons ao ----
result.bk_multi.N10.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N10,N = 10)
result.bk_multi.N50.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N50,N = 50)
result.bk_multi.N100.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N100,N = 100)
result.bk_multi.N500.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N500,N = 500)
result.bk_multi.N1000.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N1000,N = 1000)
result.bk_multi.N10000.k4 <- NN_bk_multi__hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = N10000,N = 10000)

result.bk_multi.N10.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N10,N = 10)
result.bk_multi.N50.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N50,N = 50)
result.bk_multi.N100.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N100,N = 100)
result.bk_multi.N500.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N500,N = 500)
result.bk_multi.N1000.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N1000,N = 1000)
result.bk_multi.N10000.k100 <- NN_bk_multi__hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = N10000,N = 10000)

bk_multi_hidden_layer_ao <- list(result.bk_multi.N10.k4, result.bk_multi.N50.k4,result.bk_multi.N100.k4,result.bk_multi.N500.k4,result.bk_multi.N1000.k4,result.bk_multi.N10000.k4,
                             result.bk_multi.N10.k100, result.bk_multi.N50.k100,result.bk_multi.N100.k100,result.bk_multi.N500.k100,result.bk_multi.N1000.k100,result.bk_multi.N10000.k100)
saveRDS(bk_multi_hidden_layer_ao,file = 'bk_multi_hidden_layer_ao')

#####

#Investigation 2: Equilateral, Isosceles, Scalene----
dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE", recursive = TRUE)

#####
#Neural Network 1 - 1 hidden layer shape ----
NN_1_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  bookstein_input <- samp
  
  traindata <- bookstein_input$data
  testdata <- bookstein.test$data
  
  trainlabels <- bookstein_input$shape
  testlabels <- bookstein.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
    layer_dense(units = 3,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[-which(pred == testlabels),]
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-single-large-notrain.pdf'))
  
  #Plot 1 
  plot(correctpred,asp = 1,pch = 20,col = 'blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(h = 0,lty = 3,lwd = 2.5)
  
  dev.off()
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-single-close-notrain.pdf'))
  
  #Plot 3
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  
  dev.off()
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-single-close-train.pdf'))
  
  #Plot 5
  plot(incorrectpred,asp = 1,pch = 20,col = 'red',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Single Layer, Neurons =',k))
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  points(correctpred,pch = 20, col = 'blue')
  points(traindata,pch = 21, col = 'black',bg = 'green')
  
  dev.off()
  
  distance.from.border <- apply(incorrectpred,MARGIN = 1, FUN = function(x) calc_dist_from_border_shape(x))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

#Results for 1 hidden layer different neurons shape----
result.bk.M10.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M10,N = 10)
result.bk.M50.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M50,N = 50)
result.bk.M100.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M100,N = 100)
result.bk.M500.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M500,N = 500)
result.bk.M1000.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M1000,N = 1000)
result.bk.M10000.k4 <- NN_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M10000,N = 10000)

result.bk.M10.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M10,N = 10)
result.bk.M50.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M50,N = 50)
result.bk.M100.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M100,N = 100)
result.bk.M500.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M500,N = 500)
result.bk.M1000.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M1000,N = 1000)
result.bk.M10000.k100 <- NN_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M10000,N = 10000)

bk_1_hidden_layer_shape <- list(result.bk.M10.k4, result.bk.M50.k4,result.bk.M100.k4,result.bk.M500.k4,result.bk.M1000.k4,result.bk.M10000.k4,
                             result.bk.M10.k100, result.bk.M50.k100,result.bk.M100.k100,result.bk.M500.k100,result.bk.M1000.k100,result.bk.M10000.k100)
saveRDS(bk_1_hidden_layer_shape,file = 'bk_1_hidden_layer_shape')

#####
#Neural Network 2 - multiple hidden layers----
NN_multi_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  bookstein_input <- samp
  
  traindata <- bookstein_input$data
  testdata <- bookstein.test$data 
  
  trainlabels <- bookstein_input$shape
  testlabels <- bookstein.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
    layer_dense(units = k,activation = 'relu') %>%
    layer_dense(units = 3,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[-which(pred == testlabels),]
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-multi-large-notrain.pdf'))
  
  #Plot 1 
  plot(correctpred,asp = 1,pch = 20,col = 'blue',
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Multiple Layer, Neurons =',k))
  points(incorrectpred,pch = 20, col = 'red')
  abline(h = 0,lty = 3,lwd = 2.5)
  
  dev.off()
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-multi-close-notrain.pdf'))
  
  #Plot 3
  plot(correctpred,asp = 1,pch = 20,col = 'blue',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Multiple Layer, Neurons =',k))
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  points(incorrectpred,pch = 20, col = 'red')
  
  dev.off()
  
  pdf(file = paste0("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Bookstein SHAPE/shape-bk-",N,'-',k,'-multi-close-train.pdf'))
  
  #Plot 5
  plot(incorrectpred,asp = 1,pch = 20,col = 'red',xlim = c(-2,2), ylim = c(-0.5,3),
       xlab = TeX('$U^B$'),ylab = TeX('$V^B$'),main = paste('Training size = ',N,', Network = Multiple Layer, Neurons =',k))
  points(correctpred,pch = 20, col = 'blue')
  abline(v = 0,lty = 5,lwd = 2.5)
  abline(h = 0,lty = 3,lwd = 2.5)
  xlcirc <- seq(-1.5,0.5, length.out = 1000)
  ylcirc <- sqrt(1 - (xlcirc + 0.5)**2)
  lines(xlcirc,ylcirc,lty = 5,lwd = 3)
  xrcirc <- seq(-0.5,1.5, length.out = 1000)
  yrcirc <- sqrt(1 - (xrcirc - 0.5)**2)
  lines(xrcirc,yrcirc,lty = 5,lwd = 3)
  points(traindata,pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  distance.from.border <- apply(incorrectpred,MARGIN = 1, FUN = function(x) calc_dist_from_border_shape(x))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}
#Results for multi hidden layers shape ----
result.bk.multi.M10.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M10,N = 10)
result.bk.multi.M50.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M50,N = 50)
result.bk.multi.M100.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M100,N = 100)
result.bk.multi.M500.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M500,N = 500)
result.bk.multi.M1000.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M1000,N = 1000)
result.bk.multi.M10000.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = M10000,N = 10000)

result.bk.multi.M10.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M10,N = 10)
result.bk.multi.M50.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M50,N = 50)
result.bk.multi.M100.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M100,N = 100)
result.bk.multi.M500.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M500,N = 500)
result.bk.multi.M1000.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M1000,N = 1000)
result.bk.multi.M10000.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = M10000,N = 10000)

bk_multi_hidden_layer_shape <- list(result.bk.multi.M10.k4, result.bk.multi.M50.k4,result.bk.multi.M100.k4,result.bk.multi.M500.k4,result.bk.multi.M1000.k4,result.bk.multi.M10000.k4,
                                result.bk.multi.M10.k100, result.bk.multi.M50.k100,result.bk.multi.M100.k100,result.bk.multi.M500.k100,result.bk.multi.M1000.k100,result.bk.multi.M10000.k100)
saveRDS(bk_multi_hidden_layer_shape,file = 'bk_multi_hidden_layer_shape')




