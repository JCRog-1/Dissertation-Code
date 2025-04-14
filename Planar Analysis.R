library(keras)
library(shapes)
library(latex2exp)

#This file should be run in tandem with large_dataset.R,

#Investigation 1: Acute or Obtuse ----
dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar AO", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_planar_1_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  planar_input <- planar.conversion(samp)
  
  traindata <- planar_input$planar
  testdata <- planar.test$planar
  
  trainlabels <- planar_input$ao
  testlabels <- planar.test$ao
  
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
  incorrectpred <- testdata[which(pred != testlabels),]
  x <- seq(0,1,length.out = 100)
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar AO/ao-planar-single-",N,'-',k,'-notrain.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  lines(x,sqrt(1-x**2),lwd = '2')
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar AO/ao-planar-single-",N,'-',k,'-train.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  points(traindata[,1],traindata[,2],col = 'black',bg = 'green',pch = 21)
  lines(x,sqrt(1-x**2),lwd = '2')
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  distance.from.border <- planar_dist_from_border_ao(which(pred != testlabels))
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.planar.N10.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.planar.N50.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.planar.N100.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.planar.N500.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.planar.N1000.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.planar.N10000.k4 <- NN_planar_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.planar.N10.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.planar.N50.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.planar.N100.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.planar.N500.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.planar.N1000.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.planar.N10000.k100 <- NN_planar_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000) 

planar_1_hidden_layer_ao <- list(result.planar.N10.k4, result.planar.N50.k4,result.planar.N100.k4,result.planar.N500.k4,result.planar.N1000.k4,result.planar.N10000.k4,
                               result.planar.N10.k100, result.planar.N50.k100,result.planar.N100.k100,result.planar.N500.k100,result.planar.N1000.k100,result.planar.N10000.k100)
saveRDS(planar_1_hidden_layer_ao,file = 'planar_1_hidden_layer_ao')

#Multi Hidden layer 
#A function that runs a two layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_planar_multi_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  planar_input <- planar.conversion(samp)
  
  traindata <- planar_input$planar
  testdata <-  planar.test$planar
  
  trainlabels <- planar_input$ao
  testlabels <- planar.test$ao
  
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
  incorrectpred <- testdata[which(pred != testlabels),]
  x <- seq(0,1,length.out = 100)
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar AO/ao-planar-multi-",N,'-',k,'-train.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  points(traindata[,1],traindata[,2],col = 'black',bg = 'green',pch = 21)
  lines(x,sqrt(1-x**2),lwd = '2')
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar AO/ao-planar-multi-",N,'-',k,'-notrain.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  lines(x,sqrt(1-x**2),lwd = '2')
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  distance.from.border <- planar_dist_from_border_ao(which(pred != testlabels))
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.planar_multi.N10.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.planar_multi.N50.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.planar_multi.N100.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.planar_multi.N500.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.planar_multi.N1000.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.planar_multi.N10000.k4 <- NN_planar_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.planar_multi.N10.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.planar_multi.N50.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.planar_multi.N100.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.planar_multi.N500.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.planar_multi.N1000.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.planar_multi.N10000.k100 <- NN_planar_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

planar_multi_hidden_layer_ao <- list(result.planar_multi.N10.k4, result.planar_multi.N50.k4,result.planar_multi.N100.k4,result.planar_multi.N500.k4,result.planar_multi.N1000.k4,result.planar_multi.N10000.k4,
                                   result.planar_multi.N10.k100, result.planar_multi.N50.k100,result.planar_multi.N100.k100,result.planar_multi.N500.k100,result.planar_multi.N1000.k100,result.planar_multi.N10000.k100)
saveRDS(planar_multi_hidden_layer_ao,file = 'planar_multi_hidden_layer_ao')

#Investigation 2 - Shape -----

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar Shape", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_planar_1_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  planar_input <- planar.conversion(samp)
  
  traindata <- planar_input$planar
  testdata <- planar.test$planar
  
  trainlabels <- planar_input$shape
  testlabels <- planar.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  x <- seq(0,1,length.out = 100)
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar Shape/shape-planar-single-",N,'-',k,'-notrain.pdf'))

  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar Shape/shape-planar-single-",N,'-',k,'-train.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  points(traindata[,1],traindata[,2],col = 'black',bg = 'green',pch = 21)
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  distance.from.border <- planar_dist_from_border_shape(which(pred != testlabels))
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.planar.M10.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.planar.M50.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.planar.M100.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.planar.M500.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.planar.M1000.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.planar.M10000.k4 <- NN_planar_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.planar.M10.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.planar.M50.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.planar.M100.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.planar.M500.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.planar.M1000.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.planar.M10000.k100 <- NN_planar_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000) 

planar_1_hidden_layer_shape <- list(result.planar.M10.k4, result.planar.M50.k4,result.planar.M100.k4,result.planar.M500.k4,result.planar.M1000.k4,result.planar.M10000.k4,
                                  result.planar.M10.k100, result.planar.M50.k100,result.planar.M100.k100,result.planar.M500.k100,result.planar.M1000.k100,result.planar.M10000.k100)
saveRDS(planar_1_hidden_layer_shape,file = 'planar_1_hidden_layer_shape')

#Multi Hidden layer 
#A function that runs a two layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_planar_multi_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  planar_input <- planar.conversion(samp)
  
  traindata <- planar_input$planar
  testdata <- planar.test$planar
  
  trainlabels <- planar_input$shape
  testlabels <- planar.test$shape
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
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
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  x <- seq(0,1,length.out = 100)
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar Shape/shape-planar-multi-",N,'-',k,'-notrain.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/planar Shape/shape-planar-multi-",N,'-',k,'-train.pdf'))
  
  plot(correctpred[,1],correctpred[,2],col = 'blue',pch = 20,asp = 1,xlim = c(0,1),ylim = c(0,1),xlab = 'x',ylab = 'y',
       main = paste('Training size = ',N,', Network = Multi Layer, Neurons =',k))
  points(incorrectpred[,1],incorrectpred[,2],col = 'red',pch = 20)
  points(traindata[,1],traindata[,2],col = 'black',bg = 'green',pch = 21)
  lines(x,x,lwd = '2')
  lines(x,1-x,lwd = '2')
  lines(rep(1,100),x,lwd = '2')
  
  dev.off()
  
  distance.from.border <- planar_dist_from_border_shape(which(pred != testlabels))
  
  correctpred <- which(pred == testlabels)
  incorrectpred <- which(pred != testlabels)
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred, 
              distance = distance.from.border))
}

result.planar_multi.M10.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.planar_multi.M50.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.planar_multi.M100.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.planar_multi.M500.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.planar_multi.M1000.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.planar_multi.M10000.k4 <- NN_planar_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.planar_multi.M10.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.planar_multi.M50.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.planar_multi.M100.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.planar_multi.M500.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.planar_multi.M1000.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.planar_multi.M10000.k100 <- NN_planar_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

planar_multi_hidden_layer_shape <- list(result.planar_multi.M10.k4, result.planar_multi.M50.k4,result.planar_multi.M100.k4,result.planar_multi.M500.k4,result.planar_multi.M1000.k4,result.planar_multi.M10000.k4,
                                      result.planar_multi.M10.k100, result.planar_multi.M50.k100,result.planar_multi.M100.k100,result.planar_multi.M500.k100,result.planar_multi.M1000.k100,result.planar_multi.M10000.k100)
saveRDS(planar_multi_hidden_layer_shape,file = 'planar_multi_hidden_layer_shape')
