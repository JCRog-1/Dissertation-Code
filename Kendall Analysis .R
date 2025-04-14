library(keras)
library(shapes)
library(latex2exp)
library(rgl)
library(scatterplot3d)
library(plot3Drgl)

#This file should be run in tandem with large_dataset.R, there are functions required to be in the global environment in order for this file to run

#Investigation 1: Acute or Obtuse ----

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall AO", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_kd_1_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  kendall_input <- kd.conversion(samp)
  
  traindata <- kendall_input$polar
  testdata <- kendall.test$polar
  
  trainlabels <- kendall_input$ao
  testlabels <- kendall.test$ao
  
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
  
  correctpred <- kendall.test$cart[which(pred == testlabels),]
  incorrectpred <- kendall.test$cart[-which(pred == testlabels),]

  #Can be uncommented to generate 3D plots 
  # #Plot 1 - 3D
  # open3d()
  # plot3d(correctpred[,1],correctpred[,2],correctpred[,3],col = 'blue')
  # points3d(incorrectpred[,1],incorrectpred[,2],incorrectpred[,3],col = 'red')
  # points3d(kendall_input$cart[,1],kendall_input$cart[,2],kendall_input$cart[,3],col = 'green',pch = 24)
  # points3d(x1,y1,z1,col = 'black')
  # points3d(x2,y2,z2,col = 'black')
  # points3d(x3,y3,z3,col = 'black')
  # aspect3d(1,1,0.5)
  # highlevel()
  
 
  #Plot 2 - Stereograph - with training data
  X_correct <- correctpred[,1]/(1-correctpred[,3])
  Y_correct <- correctpred[,2]/(1-correctpred[,3])
  X_incorrect <- incorrectpred[,1]/(1-incorrectpred[,3])
  Y_incorrect <- incorrectpred[,2]/(1-incorrectpred[,3])
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall AO/ao-kd-single-",N,'-',k,'-train.pdf'))
  
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,'Network = Single Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  lines(X_border[1:1000],Y_border[1:1000],col = 'black',lwd = 2)
  lines(X_border[1001:2000],Y_border[1001:2000],col = 'black',lwd = 2)
  lines(X_border[2001:3000],Y_border[2001:3000],col = 'black',lwd = 2)
  lines(X_border[3001:4000],Y_border[3001:4000],col = 'black',lwd = 2)
  lines(X_border[4001:5000],Y_border[4001:5000],col = 'black',lwd = 2)
  lines(X_border[5001:6000],Y_border[5001:6000],col = 'black',lwd = 2)
  points(kendall_input$cart[,1]/(1-kendall_input$cart[,3]),kendall_input$cart[,2]/(1-kendall_input$cart[,3]),pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall AO/ao-kd-single-",N,'-',k,'-notrain.pdf'))
  
  #plot 3 - stereograph without training data 
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Single Layer, Neurons =',k),
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
              distances = distance.from.border))
}

result.kd.N10.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.kd.N50.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.kd.N100.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.kd.N500.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.kd.N1000.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.kd.N10000.k4 <- NN_kd_1_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.kd.N10.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.kd.N50.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.kd.N100.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.kd.N500.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.kd.N1000.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.kd.N10000.k100 <- NN_kd_1_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

kd_1_hidden_layer_ao <- list(result.kd.N10.k4, result.kd.N50.k4,result.kd.N100.k4,result.kd.N500.k4,result.kd.N1000.k4,result.kd.N10000.k4,
                                result.kd.N10.k100, result.kd.N50.k100,result.kd.N100.k100,result.kd.N500.k100,result.kd.N1000.k100,result.kd.N10000.k100)
saveRDS(kd_1_hidden_layer_ao,file = 'kd_1_hidden_layer_ao')

#Neural Network 2 - multiple hidden layers
#A function that runs a two hidden layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_kd_multi_hidden_layer_ao <- function(k,epoch,batch,samp,N){
  kendall_input <- kd.conversion(samp)
  
  traindata <- kendall_input$polar
  testdata <- kendall.test$polar
  
  trainlabels <- kendall_input$ao
  testlabels <- kendall.test$ao
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(2)) %>%
    layer_dense(units = k,activation = 'relu') %>%
    layer_dense(units = 2,activation = 'sigmoid')
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  network %>% fit(traindata,to_categorical(trainlabels),epochs = epoch,batch_size = batch)
  metrics <- network %>% evaluate(testdata,to_categorical(testlabels))
  pred <- network %>% predict(testdata) %>% k_argmax()
  pred <- matrix(as.array(pred),dim(testdata)[1],1)
  
  correctpred <- kendall.test$cart[which(pred == testlabels),]
  incorrectpred <- kendall.test$cart[-which(pred == testlabels),]

  #Can be uncommented to generate 3D plots 
  #Plot 1 - 3D
  # open3d()
  # plot3d(correctpred[,1],correctpred[,2],correctpred[,3],col = 'blue')
  # points3d(incorrectpred[,1],incorrectpred[,2],incorrectpred[,3],col = 'red')
  # points3d(kendall_input$cart[,1],kendall_input$cart[,2],kendall_input$cart[,3],col = 'green',pch = 24)
  # points3d(x1,y1,z1,col = 'black')
  # points3d(x2,y2,z2,col = 'black')
  # points3d(x3,y3,z3,col = 'black')
  # aspect3d(1,1,0.5)
  # highlevel()
  
  #Plot 2 - Stereograph - with training data
  X_correct <- correctpred[,1]/(1-correctpred[,3])
  Y_correct <- correctpred[,2]/(1-correctpred[,3])
  X_incorrect <- incorrectpred[,1]/(1-incorrectpred[,3])
  Y_incorrect <- incorrectpred[,2]/(1-incorrectpred[,3])
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall AO/ao-kd-multi-",N,'-',k,'-train.pdf'))
  
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Multi Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  lines(X_border[1:1000],Y_border[1:1000],col = 'black',lwd = 2)
  lines(X_border[1001:2000],Y_border[1001:2000],col = 'black',lwd = 2)
  lines(X_border[2001:3000],Y_border[2001:3000],col = 'black',lwd = 2)
  lines(X_border[3001:4000],Y_border[3001:4000],col = 'black',lwd = 2)
  lines(X_border[4001:5000],Y_border[4001:5000],col = 'black',lwd = 2)
  lines(X_border[5001:6000],Y_border[5001:6000],col = 'black',lwd = 2)
  points(kendall_input$cart[,1]/(1-kendall_input$cart[,3]),kendall_input$cart[,2]/(1-kendall_input$cart[,3]),pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall AO/ao-kd-multi-",N,'-',k,'-notrain.pdf'))
  
  #plot 3 - stereograph without training data 
  plot(X_correct,Y_correct,pch = 20, col = 'blue',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Multi Layer, Neurons =',k),
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
              distances = distance.from.border))
}

result.kd_multi.N10.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10,N = 10)
result.kd_multi.N50.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n50,N = 50)
result.kd_multi.N100.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n100,N = 100)
result.kd_multi.N500.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n500,N = 500)
result.kd_multi.N1000.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.kd_multi.N10000.k4 <- NN_kd_multi_hidden_layer_ao(k = 4,epoch = 10,batch = 10,samp = n10000,N = 10000)

result.kd_multi.N10.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10,N = 10)
result.kd_multi.N50.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n50,N = 50)
result.kd_multi.N100.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n100,N = 100)
result.kd_multi.N500.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n500,N = 500)
result.kd_multi.N1000.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n1000,N = 1000)
result.kd_multi.N10000.k100 <- NN_kd_multi_hidden_layer_ao(k = 100,epoch = 10,batch = 10,samp = n10000,N = 10000)

kd_multi_hidden_layer_ao <- list(result.kd_multi.N10.k4, result.kd_multi.N50.k4,result.kd_multi.N100.k4,result.kd_multi.N500.k4,result.kd_multi.N1000.k4,result.kd_multi.N10000.k4,
                                result.kd_multi.N10.k100, result.kd_multi.N50.k100,result.kd_multi.N100.k100,result.kd_multi.N500.k100,result.kd_multi.N1000.k100,result.kd_multi.N10000.k100)
saveRDS(kd_multi_hidden_layer_ao,file = 'kd_multi_hidden_layer_ao')


#Investigation 2: Equilateral, Isosceles, Scalene ----

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall Shape", recursive = TRUE)
#A function that runs a single layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_kd_1_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  kendall_input <- kd.conversion(samp)
  
  traindata <- kendall_input$polar
  testdata <- kendall.test$polar
  
  trainlabels <- kendall_input$shape
  testlabels <- kendall.test$shape
  
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
  
  correctpred <- kendall.test$cart[which(pred == testlabels),]
  incorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  #Can be uncommented to generate 3D plots 
  # #Plot 1 - 3D
  # open3d()
  # plot3d(correctpred[,1],correctpred[,2],correctpred[,3],col = 'blue')
  # points3d(incorrectpred[,1],incorrectpred[,2],incorrectpred[,3],col = 'red')
  # points3d(kendall_input$cart[,1],kendall_input$cart[,2],kendall_input$cart[,3],col = 'green',pch = 24)
  # points3d(xx1,yy1,zz1,col = 'black')
  # points3d(xx2,yy2,zz2,col = 'black')
  # points3d(xx3,yy3,zz3,col = 'black')
  # points3d(xx4,yy4,zz4,col = 'black')
  # points3d(xx5,yy5,zz5,col = 'black')
  # points3d(xx6,yy6,zz6,col = 'black')
  # aspect3d(1,1,0.5)
  # highlevel()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall Shape/shape-kd-single-",N,'-',k,'-train.pdf'))
  
  #Plot 2 - Stereograph - with training data
  X_correct <- correctpred[,1]/(1-correctpred[,3])
  Y_correct <- correctpred[,2]/(1-correctpred[,3])
  X_incorrect <- incorrectpred[,1]/(1-incorrectpred[,3])
  Y_incorrect <- incorrectpred[,2]/(1-incorrectpred[,3])
  
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Single Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  points(kendall_input$cart[,1]/(1-kendall_input$cart[,3]),kendall_input$cart[,2]/(1-kendall_input$cart[,3]),pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall Shape/shape-kd-single-",N,'-',k,'-notrain.pdf'))
  
  #plot 3 - stereograph without training data 
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Single Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_shape(which(pred != testlabels))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred,
              distances = distance.from.border))
}

result.kd.M10.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.kd.M50.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.kd.M100.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.kd.M500.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.kd.M1000.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.kd.M10000.k4 <- NN_kd_1_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.kd.M10.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.kd.M50.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.kd.M100.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.kd.M500.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.kd.M1000.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.kd.M10000.k100 <- NN_kd_1_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

kd_1_hidden_layer_shape <- list(result.kd.M10.k4, result.kd.M50.k4,result.kd.M100.k4,result.kd.M500.k4,result.kd.M1000.k4,result.kd.M10000.k4,
                                 result.kd.M10.k100, result.kd.M50.k100,result.kd.M100.k100,result.kd.M500.k100,result.kd.M1000.k100,result.kd.M10000.k100)
saveRDS(kd_1_hidden_layer_shape,file = 'kd_1_hidden_layer_shape')

#Neural Network 2 - multiple hidden layers
#A function that runs a two hidden layer NN returns the weights, loss, accuracy, indexes of correctly and incorrectly identified points in the test set and the distances of each point from the border
NN_multi_hidden_layer_shape <- function(k,epoch,batch,samp,N){
  kendall_input <- kd.conversion(samp)
  
  traindata <- kendall_input$polar
  testdata <- kendall.test$polar
  
  trainlabels <- kendall_input$shape
  testlabels <- kendall.test$shape
  
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
  
  correctpred <- kendall.test$cart[which(pred == testlabels),]
  incorrectpred <- kendall.test$cart[-which(pred == testlabels),]
  
  #Can be uncommented to generate 3D plots 
  # #Plot 1 - 3D
  # open3d()
  # plot3d(correctpred[,1],correctpred[,2],correctpred[,3],col = 'blue')
  # points3d(incorrectpred[,1],incorrectpred[,2],incorrectpred[,3],col = 'red')
  # points3d(kendall_input$cart[,1],kendall_input$cart[,2],kendall_input$cart[,3],col = 'green',pch = 24)
  # points3d(xx1,yy1,zz1,col = 'black')
  # points3d(xx2,yy2,zz2,col = 'black')
  # points3d(xx3,yy3,zz3,col = 'black')
  # points3d(xx4,yy4,zz4,col = 'black')
  # points3d(xx5,yy5,zz5,col = 'black')
  # points3d(xx6,yy6,zz6,col = 'black')
  # aspect3d(1,1,0.5)
  # highlevel()
  
  #Plot 2 - Stereograph - with training data
  X_correct <- correctpred[,1]/(1-correctpred[,3])
  Y_correct <- correctpred[,2]/(1-correctpred[,3])
  X_incorrect <- incorrectpred[,1]/(1-incorrectpred[,3])
  Y_incorrect <- incorrectpred[,2]/(1-incorrectpred[,3])
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall Shape/shape-kd-multi-",N,'-',k,'-train.pdf'))
  
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Multi Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  points(kendall_input$cart[,1]/(1-kendall_input$cart[,3]),kendall_input$cart[,2]/(1-kendall_input$cart[,3]),pch = 21,col = 'black',bg = 'green')
  
  dev.off()
  
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Kendall Shape/shape-kd-multi-",N,'-',k,'-notrain.pdf'))
  
  #plot 3 - stereograph without training data 
  plot(XX_border,YY_border,col = 'black',asp = 1,xlab = 'x',ylab = 'y',main =  paste('Training size = ',N,', Network = Multi Layer, Neurons =',k),
       xlim = c(-0.7,0.7),ylim = c(-0.7,0.7))
  points(X_correct,Y_correct,pch = 20, col = 'blue')
  points(X_incorrect,Y_incorrect,pch = 20,col = 'red')
  
  dev.off()
  
  distance.from.border <- calc_dist_from_border_shape(which(pred != testlabels))
  
  correctpred <- testdata[which(pred == testlabels),]
  incorrectpred <- testdata[which(pred != testlabels),]
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred,
              distances = distance.from.border))
}

result.kd_multi.M10.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10,N = 10)
result.kd_multi.M50.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m50,N = 50)
result.kd_multi.M100.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m100,N = 100)
result.kd_multi.M500.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m500,N = 500)
result.kd_multi.M1000.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.kd_multi.M10000.k4 <- NN_multi_hidden_layer_shape(k = 4,epoch = 10,batch = 10,samp = m10000,N = 10000)

result.kd_multi.M10.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10,N = 10)
result.kd_multi.M50.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m50,N = 50)
result.kd_multi.M100.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m100,N = 100)
result.kd_multi.M500.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m500,N = 500)
result.kd_multi.M1000.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m1000,N = 1000)
result.kd_multi.M10000.k100 <- NN_multi_hidden_layer_shape(k = 100,epoch = 10,batch = 10,samp = m10000,N = 10000)

kd_multi_hidden_layer_shape <- list(result.kd_multi.M10.k4, result.kd_multi.M50.k4,result.kd_multi.M100.k4,result.kd_multi.M500.k4,result.kd_multi.M1000.k4,result.kd_multi.M10000.k4,
                                result.kd_multi.M10.k100, result.kd_multi.M50.k100,result.kd_multi.M100.k100,result.kd_multi.M500.k100,result.kd_multi.M1000.k100,result.kd_multi.M10000.k100)
saveRDS(kd_multi_hidden_layer_shape,file = 'kd_multi_hidden_layer_shape')





























