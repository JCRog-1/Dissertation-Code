numbers <- read.csv('dataset_32_pendigits.csv')
library(shapes)
library(keras)
library(tensorflow)
library(abind)
library(reticulate)
library(tensr)


digits <- array_reshape(as.matrix(numbers)[,2:17],dim = c(10992,8,2),order = 'C') #Extract landmars of numbers
data <- list(nums = digits,labels = numbers$X.class.) #Form list of landmarks and associated labels 

#Initialise the training data 
rand_affine_digit <- function(X){
  
  #random rotation in [0,2*pi)
  theta <- runif(1,min = 0, max = 2*pi)
  rot.mat <- matrix(c(cos(theta),sin(theta),-sin(theta),cos(theta)),2,2)
  
  #random translation so that the center of the triangle moves to anywhere in the box [-5,5] x [-5,5]
  translation <- matrix(rep(runif(2,min = -5,max = 5),4),2,8)
  
  #random scaling within (0.25,3]
  scaling <- runif(1,min = 0.01,max = 2)
  
  Y <- scaling*(rot.mat%*%X) + translation
  
  return(t(Y))
}

data.train.start <- list(nums = data$nums[1:9000,,],labels = data$labels[1:9000])
new_data.train <- array(NA,dim = c(8,2,length(data.train.start$labels)))
for (i in 1:length(data.train.start$labels)){
  new_data.train[,,i] <- data.train.start$nums[i,,]
}

data.train <- list(nums = new_data.train,labels = data.train.start$labels)

data.test.start <- list(nums = data$nums[9001:10992,,],labels = data$labels[9001:10992])

new_data.test <- array(NA,dim = c(8,2,length(data.test.start$labels)))
for (i in 1:length(data.test.start$labels)){
  new_data.test[,,i] <- data.test.start$nums[i,,]
}

augmented_test_data1 <- array_reshape(apply(data.test.start$nums,MARGIN = 1,FUN = function(x) rand_affine_digit(t(x))),dim = c(8,2,1992),order = 'F')
augmented_test_data2 <- array_reshape(apply(data.test.start$nums,MARGIN = 1,FUN = function(x) rand_affine_digit(t(x))),dim = c(8,2,1992),order = 'F')

data.test <- list(nums = abind(new_data.test,augmented_test_data1,augmented_test_data2), 
                  labels = rbind(as.matrix(data.test.start$labels),as.matrix(data.test.start$labels),as.matrix(data.test.start$labels)))

zeros <- data.train$nums[,,which(data.train$labels == 0)]
ones <- data.train$nums[,,which(data.train$labels == 1)]
twos <- data.train$nums[,,which(data.train$labels == 2)]
threes  <- data.train$nums[,,which(data.train$labels == 3)]
fours <- data.train$nums[,,which(data.train$labels == 4)]
fives <- data.train$nums[,,which(data.train$labels == 5)]
sixes <- data.train$nums[,,which(data.train$labels == 6)]
sevens <- data.train$nums[,,which(data.train$labels == 7)]
eights <- data.train$nums[,,which(data.train$labels == 8)]
nines <- data.train$nums[,,which(data.train$labels == 9)]

sorted_numbers <- list('0' = zeros,'1' = ones,'2' = twos, '3' = threes, '4' = fours, '5' = fives, '6' = sixes, '7' = sevens, '8' = eights, '9' = nines)
#####
#Template Numbers ----
zero <- matrix(c(1.47,1-1/sqrt(2),1-1/sqrt(2),1,2,2+1/sqrt(2),2+1/sqrt(2),1.53,
                 1+2/sqrt(2),1+1/sqrt(2),1/sqrt(2),0,0,1/sqrt(2),1+1/sqrt(2),1+2/sqrt(2)),8,2)

one <- matrix(c(rep(1,6),1-1/sqrt(2),1-2/sqrt(2)
                ,seq(0,5,by=1),5-1/sqrt(2),5-2/sqrt(2)),8,2)

two <- matrix(c(1,2,1+2/sqrt(2),1+1/sqrt(2),1,2,3,4,
                3,3,2,1,0,0,0,0),8,2)

three <- matrix(c(1,2,2+1/sqrt(2),2,1,2+1/sqrt(2),2,1
                  ,0,0,1/sqrt(2),2/sqrt(2),2/sqrt(2),3/sqrt(2),4/sqrt(2),4/sqrt(2)),8,2)

four <- matrix(c(1,1,1,1,0.9,-0.1,-1.1,-1.1,
                 0,1,2,3,2,2,2,3),8,2)

five <- matrix(c(1,2,2+1/sqrt(2),2+1/sqrt(2),2,1,1,2,
                 0,0,1/sqrt(2),1+1/sqrt(2),1+2/sqrt(2),1+2/sqrt(2),2+2/sqrt(2),2+2/sqrt(2)),8,2)

six <- matrix(c(0.6,2,2+1/sqrt(2),2,1,0.5,5/6,5/6+1/sqrt(2),
                sqrt(3)/2,2/sqrt(2),1/sqrt(2),0,0,sqrt(3)/2,sqrt(3)/2+2*sqrt(2)/3,sqrt(3)/2+2*sqrt(2)/3 + 1/sqrt(2)),8,2)

seven <- matrix(c(1,1.5,2,2.5,3,3.5,2.5,1.5,
                  0,sqrt(3)/2,sqrt(3),3*sqrt(3)/2,2*sqrt(3),5*sqrt(3)/2,5*sqrt(3)/2,5*sqrt(3)/2),8,2)

eight <- matrix(c(0.5,1,2,2.5,0.5,1.5,2.5,0.55,
                  sqrt(3)/2,0,0,sqrt(3)/2,sqrt(3)/2 + 1, sqrt(3)/2 + 2, sqrt(3)/2 + 1, sqrt(3)/2 +0.05),8,2)

nine <- matrix(c(1-1/sqrt(2),1,1,1,1-1/sqrt(2),1-2/sqrt(2),1-1/sqrt(2),0.95,
                 1-1/sqrt(2),1,2,3,3+1/sqrt(2),3,3-1/sqrt(2),3-1/sqrt(2)),8,2)

templates <- abind(zero,one,two,three,four,five,six,seven,eight,nine,rev.along = 3)
#####
#Population means----
mean_0 <- frechet(data.test$nums[,,which(data.test$labels == 0)],mean = 'intrinsic')$mshape
mean_1 <- frechet(data.test$nums[,,which(data.test$labels == 1)],mean = 'intrinsic')$mshape
mean_2 <- frechet(data.test$nums[,,which(data.test$labels == 2)],mean = 'intrinsic')$mshape
mean_3 <- frechet(data.test$nums[,,which(data.test$labels == 3)],mean = 'intrinsic')$mshape
mean_4 <- frechet(data.test$nums[,,which(data.test$labels == 4)],mean = 'intrinsic')$mshape
mean_5 <- frechet(data.test$nums[,,which(data.test$labels == 5)],mean = 'intrinsic')$mshape
mean_6 <- frechet(data.test$nums[,,which(data.test$labels == 6)],mean = 'intrinsic')$mshape
mean_7 <- frechet(data.test$nums[,,which(data.test$labels == 7)],mean = 'intrinsic')$mshape
mean_8 <- frechet(data.test$nums[,,which(data.test$labels == 8)],mean = 'intrinsic')$mshape
mean_9 <- frechet(data.test$nums[,,which(data.test$labels == 9)],mean = 'intrinsic')$mshape
means <- abind(mean_0,mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7,mean_8,mean_9,rev.along = 3)
saveRDS(means,file = 'mean_digit')
means <- readRDS('mean_digit')
#####
#Sample of digits----
sample_digits <- function(N){
  set <- list(nums = array(NA,dim = c(8,2,N)), labels = matrix(NA,N))
  for (i in 0:9){
    inds <- sample(dim(sorted_numbers[[paste(i)]])[3],N/10)
    set$nums[,,(i*N/10 + 1):((i+1)*N/10)] <- sorted_numbers[[paste(i)]][,,inds]
    set$labels[(i*N/10 + 1):((i+1)*N/10)] <- rep(i,N/10)
  }
  return(set)
}

set.seed(2500);D10 <- sample_digits(10)
set.seed(2501);D50 <- sample_digits(50)
set.seed(2502);D100 <- sample_digits(100)
set.seed(2503);D500 <- sample_digits(500)
set.seed(2504);D1000 <- sample_digits(1000)
set.seed(2505);D5000 <- sample_digits(5000)
#####
#LQ Decomposition on Digit Data ----
LQ.digit.conversion <- function(data_set){
  digits <- data_set$nums
  
  #Calulate the lower triangular matrix 
  L <- array_reshape(apply(digits,MARGIN = 3,FUN = function(x) lq(defh(7)%*%x)$L),
                     dim = c(7,2,dim(digits)[3]),order = 'F')
  
  W <- array_reshape(apply(L,MARGIN = 3,FUN = function(x) x/norm(x,type = 'F')),
                     dim = c(7,2,dim(digits)[3]),order = 'F')
  
  set <- list(LQ = W,label = data_set$labels)
  
  return(set)
}
LQ_input_digit <- LQ.digit.conversion(data_set = data.train)
LQ.test.digit <- LQ.digit.conversion(data_set = data.test)

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits LQ", recursive = TRUE)

NN_LQ_1_hidden_layer_digits <- function(k,epoch,batch,samp,N){
  LQ_input_digit <- LQ.digit.conversion(samp)
  
  traindata <- t(array_reshape(LQ_input_digit$LQ, dim = c(14,N), order = 'F'))
  testdata <- t(array_reshape(LQ.test.digit$LQ, dim = c(14,length(LQ.test.digit$label)), order = 'F'))
  
  trainlabels <- LQ_input_digit$label
  testlabels <- LQ.test.digit$label
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'elu',input_shape = c(14)) %>%
    layer_dense(units = 10,activation = 'softmax')
  
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
  
  mean_shapes_correct <- array(NA,dim = c(8,2,10))
  mean_shapes_incorrect <- array(NA,dim = c(8,2,10))
  correct_distance_from_template <- matrix(NA,nrow = 10)
  incorrect_distance_from_template <- matrix(NA,nrow = 10)
  distance_from_each_other <- matrix(NA,nrow = 10)
  
  #Getting data for mean shapes and distances
  for (i in 0:9){
    print(i)

    if (length(which(pred == i & testlabels == i))>0){

      if (length(which(pred == i & testlabels == i)) == 1){

        mean_shapes_correct[,,(i+1)] <- data.test$nums[,,which(pred == i & testlabels == i)]
      }
      else{mean_shapes_correct[,,(i+1)] <- frechet(data.test$nums[,,which(pred == i & testlabels == i)],mean = 'intrinsic')$mshape}

      correct_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_correct[,,(i+1)],reflect = TRUE)
    }

    if (length(which(pred != i & testlabels == i))>0){

      if (length(which(pred != i & testlabels == i)) == 1){

        mean_shapes_incorrect[,,(i+1)] <- data.test$nums[,,which(pred != i & testlabels == i)]
      }

      else{mean_shapes_incorrect[,,(i+1)] <- frechet(data.test$nums[,,which(pred != i & testlabels == i)],mean = 'intrinsic')$mshape}

      incorrect_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_incorrect[,,(i+1)],reflect = TRUE)
    }
    
    if (length(which(pred == i & testlabels == i))>0 & length(which(pred != i & testlabels == i))>0){
      distance_from_each_other[i+1] <- riemdist(mean_shapes_correct[,,(i+1)],mean_shapes_incorrect[,,(i+1)],reflect = TRUE)
    }
  }
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred,mean_shape_right = mean_shapes_correct,
              mean_shape_wrong = mean_shapes_incorrect,distances_correct = correct_distance_from_template,
              distances_incorrect = incorrect_distance_from_template ))
}

result.LQ.D10.k16 <- NN_LQ_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D10,N = 10)
result.LQ.D100.k16 <- NN_LQ_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D100,N = 100)
result.LQ.D1000.k16 <- NN_LQ_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.LQ.D5000.k16 <- NN_LQ_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D5000,N = 5000)

result.LQ.D10.k100 <- NN_LQ_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D10,N = 10)
result.LQ.D100.k100 <- NN_LQ_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D100,N = 100)
result.LQ.D1000.k100 <- NN_LQ_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.LQ.D5000.k100 <- NN_LQ_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D5000,N = 5000)

generate_plots_LQ <- function(result,N,k){
  #pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits LQ/single-",N,'-',k,'.pdf'))
  par(mfrow = c(3,10))
  for (i in 0:9){
    
    if (i != 3){
      plotshapes(means[i+1,,],joinline = c(1:8))
      
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
      
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
    
    if (i == 3){
      plotshapes(means[i+1,,],joinline = c(1,2,3,4,5,4,6,7,8))
      
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
      
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
  }
  #dev.off()
}

generate_plots_LQ(result.LQ.D10.k16,10,16)
generate_plots_LQ(result.LQ.D100.k16,100,16)
generate_plots_LQ(result.LQ.D1000.k16,1000,16)
generate_plots_LQ(result.LQ.D5000.k16,5000,16)
generate_plots_LQ(result.LQ.D10.k100,10,100)
generate_plots_LQ(result.LQ.D100.k100,100,100)
generate_plots_LQ(result.LQ.D1000.k100,1000,100)
generate_plots_LQ(result.LQ.D5000.k100,5000,100)

LQ_1_hidden_layer_digit <- list(result.LQ.D10.k16,result.LQ.D100.k16,result.LQ.D1000.k16,result.LQ.D5000.k16,
                                result.LQ.D10.k100,result.LQ.D100.k100,result.LQ.D1000.k100,result.LQ.D5000.k100)
saveRDS(LQ_1_hidden_layer_digit,file = 'LQ_1_hidden_layer_digit')


#####
#Raw Data Coordinates on Digit Data ----

NN_raw_1_hidden_layer_digits <- function(k,epoch,batch,samp,N){
  
  traindata <- (array_reshape(samp$nums, dim = c(N,16), order = 'F'))
  testdata <- t(array_reshape(data.test$nums, dim = c(16,length(data.test$labels)), order = 'F'))
  
  trainlabels <- samp$labels
  testlabels <- data.test$labels
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(16)) %>%
    layer_dense(units = 10,activation = 'softmax')
  
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
  
  mean_shapes_correct <- array(NA,dim = c(8,2,10))
  mean_shapes_incorrect <- array(NA,dim = c(8,2,10))
  correct_distance_from_template <- matrix(NA,nrow = 10)
  incorrect_distance_from_template <- matrix(NA,nrow = 10)
  distance_from_each_other <- matrix(NA,nrow = 10)
  #Getting data for mean shapes and distances
  for (i in 0:9){
     print(i)
     if (length(which(pred == i & testlabels == i))>0){
  
       if (length(which(pred == i & testlabels == i)) == 1){
  
         mean_shapes_correct[,,(i+1)] <- data.test$nums[,,which(pred == i & testlabels == i)]
       }
       else{mean_shapes_correct[,,(i+1)] <- frechet(data.test$nums[,,which(pred == i & testlabels == i)],mean = 'intrinsic')$mshape}
  
       correct_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_correct[,,(i+1)])
     }
  
     if (length(which(pred != i & testlabels == i))>0){
  
       if (length(which(pred != i & testlabels == i)) == 1){
  
         mean_shapes_incorrect[,,(i+1)] <- data.test$nums[,,which(pred != i & testlabels == i)]
       }
  
       else{mean_shapes_incorrect[,,(i+1)] <- frechet(data.test$nums[,,which(pred != i & testlabels == i)],mean = 'intrinsic')$mshape}
  
       incorrect_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_incorrect[,,(i+1)])
     }
     if (length(which(pred == i & testlabels == i))>0 & length(which(pred != i & testlabels == i))>0){
       distance_from_each_other[i+1] <- riemdist(mean_shapes_correct[,,(i+1)],mean_shapes_incorrect[,,(i+1)],reflect = TRUE)
     }  
   }
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred,mean_shape_right = mean_shapes_correct,
              mean_shape_wrong = mean_shapes_incorrect,distances_correct = correct_distance_from_template,
              distances_incorrect = incorrect_distance_from_template,distance_between = distance_from_each_other ))
}

result.raw.D10.k16 <- NN_raw_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D10,N = 10)
result.raw.D100.k16 <- NN_raw_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D100,N = 100)
result.raw.D1000.k16 <- NN_raw_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.raw.D5000.k16 <- NN_raw_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D5000,N = 5000)

result.raw.D10.k100 <- NN_raw_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D10,N = 10)
result.raw.D100.k100 <- NN_raw_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D100,N = 100)
result.raw.D1000.k100 <- NN_raw_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.raw.D5000.k100 <- NN_raw_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D5000,N = 5000)

dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits Raw", recursive = TRUE)

generate_plots_raw <- function(result,N,k){
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits Raw/single-",N,'-',k,'.pdf'))
  par(mfrow = c(3,3))
  for (i in 0:9){
    
    if (i != 3){
      plotshapes(means[i+1,,],joinline = c(1:8))
    
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
        
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
    
    if (i == 3){
      plotshapes(means[i+1,,],joinline = c(1,2,3,4,5,4,6,7,8))
      
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
      
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
  }
  dev.off()
}
generate_plots_raw(result.raw.D10.k16,10,16)
generate_plots_raw(result.raw.D100.k16,100,16)
generate_plots_raw(result.raw.D1000.k16,1000,16)
generate_plots_raw(result.raw.D5000.k16,5000,16)
generate_plots_raw(result.raw.D10.k100,10,100)
generate_plots_raw(result.raw.D100.k100,100,100)
generate_plots_raw(result.raw.D1000.k100,1000,100)
generate_plots_raw(result.raw.D5000.k100,5000,100)

raw_1_hidden_layer_digit <- list(result.raw.D10.k16,result.raw.D100.k16,result.raw.D1000.k16,result.raw.D5000.k16,
                                result.raw.D10.k100,result.raw.D100.k100,result.raw.D1000.k100,result.raw.D5000.k100)
saveRDS(raw_1_hidden_layer_digit,file = 'raw_1_hidden_layer_digit')
#####
#Raw Data Coordinates on Digit Data with DA----

NN_rawDA_1_hidden_layer_digits <- function(k,epoch,batch,samp,N){
  
  augmented_train_data1 <- array_reshape(apply(samp$nums,MARGIN = 3,FUN = function(x) rand_affine_digit(t(x))),dim = c(8,2,length(samp$labels)),order = 'F')
  augmented_train_data2 <- array_reshape(apply(samp$nums,MARGIN = 3,FUN = function(x) rand_affine_digit(t(x))),dim = c(8,2,length(samp$labels)),order = 'F')
  trainDA <- abind(samp$nums,augmented_train_data1,augmented_train_data2)
  
  traindata <- (array_reshape(trainDA, dim = c(3*N,16), order = 'F'))
  testdata <- t(array_reshape(data.test$nums, dim = c(16,length(data.test$labels)), order = 'F'))
  
  trainlabels <- rbind(samp$labels,samp$labels,samp$labels)
  testlabels <- data.test$labels
  
  network <- keras_model_sequential() %>%
    layer_dense(units = k,activation = 'relu',input_shape = c(16)) %>%
    layer_dense(units = 10,activation = 'softmax')
  
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
  
  mean_shapes_correct <- array(NA,dim = c(8,2,10))
  mean_shapes_incorrect <- array(NA,dim = c(8,2,10))
  correct_distance_from_template <- matrix(NA,nrow = 10)
  incorrect_distance_from_template <- matrix(NA,nrow = 10)
  distance_from_each_other <- matrix(NA,nrow = 10)
  #Getting data for mean shapes and distances 
  for (i in 0:9){
    print(i)
    if (length(which(pred == i & testlabels == i))>0){
      
      if (length(which(pred == i & testlabels == i)) == 1){
        
        mean_shapes_correct[,,(i+1)] <- data.test$nums[,,which(pred == i & testlabels == i)]
      } 
      else{mean_shapes_correct[,,(i+1)] <- frechet(data.test$nums[,,which(pred == i & testlabels == i)],mean = 'intrinsic')$mshape}
      
      correct_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_correct[,,(i+1)])
    }
    
    if (length(which(pred != i & testlabels == i))>0){
      
      if (length(which(pred != i & testlabels == i)) == 1){
        
        mean_shapes_incorrect[,,(i+1)] <- data.test$nums[,,which(pred != i & testlabels == i)]
      }
      
      else{mean_shapes_incorrect[,,(i+1)] <- frechet(data.test$nums[,,which(pred != i & testlabels == i)],mean = 'intrinsic')$mshape}
      
      incorrect_distance_from_template[i+1] <- riemdist(means[(i+1),,],mean_shapes_incorrect[,,(i+1)])
    }
    if (length(which(pred == i & testlabels == i))>0 & length(which(pred != i & testlabels == i))>0){
      distance_from_each_other[i+1] <- riemdist(mean_shapes_correct[,,(i+1)],mean_shapes_incorrect[,,(i+1)],reflect = TRUE)
    }  
    
  }
  
  return(list(weights = get_weights(network),met = metrics, right = correctpred, wrong = incorrectpred,mean_shape_right = mean_shapes_correct,
              mean_shape_wrong = mean_shapes_incorrect,distances_correct = correct_distance_from_template,
              distances_incorrect = incorrect_distance_from_template,distance_between = distance_from_each_other  ))
}

result.rawDA.D10.k16 <- NN_rawDA_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D10,N = 10)
result.rawDA.D100.k16 <- NN_rawDA_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D100,N = 100)
result.rawDA.D1000.k16 <- NN_rawDA_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.rawDA.D5000.k16 <- NN_rawDA_1_hidden_layer_digits(k = 16,epoch = 30,batch = 10,samp = D5000,N = 5000)
result.rawDA.D10.k100 <- NN_rawDA_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D10,N = 10)
result.rawDA.D100.k100 <- NN_rawDA_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D100,N = 100)
result.rawDA.D1000.k100 <- NN_rawDA_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D1000,N = 1000)
result.rawDA.D5000.k100 <- NN_rawDA_1_hidden_layer_digits(k = 100,epoch = 30,batch = 10,samp = D5000,N = 5000)

rawDA_1_hidden_layer_digit <- list(result.rawDA.D10.k16,result.rawDA.D100.k16,result.rawDA.D1000.k16,result.rawDA.D5000.k16,
                                 result.rawDA.D10.k100,result.rawDA.D100.k100,result.rawDA.D1000.k100,result.rawDA.D5000.k100)
saveRDS(rawDA_1_hidden_layer_digit,file = 'rawDA_1_hidden_layer_digit')


dir.create("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits Raw DA", recursive = TRUE)

generate_plots_rawDA <- function(result,N,k){
  pdf(file = paste("/Users/jackrogers/Documents/Year 4/Images for Dissertation/Digits Raw DA/single-",N,'-',k,'.pdf'))
  par(mfrow = c(3,3))
  for (i in 0:9){
    
    if (i != 3){
      plotshapes(means[i+1,,],joinline = c(1:8))
      
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
      
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
    
    if (i == 3){
      plotshapes(means[i+1,,],joinline = c(1,2,3,4,5,4,6,7,8))
      
      if (is.na(result$mean_shape_right[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_right[,,i+1],joinline = c(1:8))}
      
      if (is.na(result$mean_shape_wrong[1,1,i+1])){
        plot(0,0,xlab = '',ylab = '')
      }
      else{plotshapes(result$mean_shape_wrong[,,i+1],joinline = c(1:8))}
    }
  }
  dev.off()
}
generate_plots_rawDA(result.rawDA.D10.k16,10,16)
generate_plots_rawDA(result.rawDA.D100.k16,100,16)
generate_plots_rawDA(result.rawDA.D1000.k16,1000,16)
generate_plots_rawDA(result.rawDA.D5000.k16,5000,16)
generate_plots_rawDA(result.rawDA.D10.k100,10,100)
generate_plots_rawDA(result.rawDA.D100.k100,100,100)
generate_plots_rawDA(result.rawDA.D1000.k100,1000,100)
generate_plots_rawDA(result.rawDA.D5000.k100,5000,100)
#####












