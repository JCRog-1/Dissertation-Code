library(shapes)
library(plot3Drgl)
library(abind)
library(keras)
library(parallel)
library(tensr)

#A function that takes an 2x3 matrix and applies a random similarity transformation
rand_affine <- function(X){
  
  #random rotation in [0,2*pi)
  theta <- runif(1,min = 0, max = 2*pi)
  rot.mat <- matrix(c(cos(theta),sin(theta),-sin(theta),cos(theta)),2,2)
  
  #random translation so that the center of the triangle moves to anywhere in the box [-5,5] x [-5,5]
  translation <- matrix(rep(runif(2,min = -5,max = 5),3),2,3)
  
  #random scaling within (0.25,3]
  scaling <- runif(1,min = 0.01,max = 2)
  
  Y <- scaling*(rot.mat%*%X) + translation
  
  return(Y)
}

#Generate equilateral triangles
equilateral_tri <- function(){
  H <- defh(2)
  new_tri <- rand_affine(H)
  return(new_tri)
}

#Generate isosceles triangles
isos_tri <- function(){
  peak <- runif(2,min = -5,max = 5 )
  angle <- runif(1,min = 0,max = 2*pi)
  base_length <- sqrt(2 - 2*cos(angle))
  height <- sqrt(1 - (base_length/2)**2)
  
  tri1 <- matrix(c(peak[1],peak[2],peak[1] - 0.5*base_length,peak[2]-height,
                  peak[1] + 0.5*base_length,peak[2]-height),2,3)
  
  tri3 <- matrix(c(-0.5,0,0.5,0,0,rnorm(1)),2,3)
  
  c <- sample(3,1)
  if (c == 1){new_tri <- rand_affine(tri1)}
  if (c == 2){
    new_tri <- rand_affine(tri1)
    new_tri[1,] <- -new_tri[1,]
    }
  else{new_tri <- rand_affine(tri3)}
  
  return(new_tri)
}

#Generate scalene triangles
scalene_tri <- function(){
  tri <- matrix(rnorm(6),2,3)
  new_tri <- rand_affine(tri)
  return(new_tri)
}

#Determines if a triangle is acute or obtuse 
ao.identifier <- function(X){
  a <- norm(X[,1] - X[,2],type = '2')**2
  b <- norm(X[,1] - X[,3],type = '2')**2 
  c <- norm(X[,3] - X[,2],type = '2')**2 
  
  if (max(a,b,c) < (a + b + c - max(a,b,c))){return(1)}
  
  else{return(0)}
}

#Uses the above functions to generate N random triangles, by design 1/6 will be equilateral, 1/6 isosceles, 2/3 scalene
triangle_function <- function(N){
  data <- array(NA,dim = c(3,2,N))
  ao.label <- matrix(NA,N,1)
  shape.label <- matrix(NA,N,1)
  
  for(i in 1:N){
    j <- sample(6,1)
    
    if (j == 1){
      tri <- equilateral_tri()
      data[,,i] <- t(tri)
      ao.label[i] <- 1
      shape.label[i] <- 0
    }
    
    if (j == 2){
      tri <- isos_tri()
      data[,,i] <- t(tri)
      ao.label[i] <- ao.identifier(tri)
      shape.label[i] <- 1
    }
    
    if (j %in% 3:6){
      tri <- scalene_tri()
      data[,,i] <- t(tri)
      ao.label[i] <- ao.identifier(tri)
      shape.label[i] <- 2
    }
  }
  
  set <- list(triangles = data,ao = ao.label,shape = shape.label)
  
  return(set)
}

#Ensures same triangles are generated every time its ran
#Both overall training and testing set
set.seed(1); triangle_data_train <- triangle_function(100000)
set.seed(2); triangle_data_test <- triangle_function(3000)

#We use these ahead to sample equally from acute and obtuse triangles
triangle_data_acute <- triangle_data_train$triangles[,,which(triangle_data_train$ao == 1)]
triangle_data_obtuse <- triangle_data_train$triangles[,,which(triangle_data_train$ao == 0)]
dim(triangle_data_isosceles)

#N must be even
#Generate an N sample of acute and obtuse triangles from the data set
sampling_ao <- function(N){
  k <- sample(min(dim(triangle_data_acute)[3],dim(triangle_data_obtuse)[3]))[1:(N/2)]
  acute_sample <- triangle_data_acute[,,k]
  obtuse_sample <- triangle_data_obtuse[,,k]
  
  data <- abind(acute_sample,obtuse_sample)
  label <- rbind(matrix(rep(1,N/2),ncol = 1),matrix(rep(0,N/2),ncol = 1))
  shapes <- rbind(matrix(triangle_data_train$shape[which(triangle_data_train$ao == 1)][k],ncol = 1),
                 matrix(triangle_data_train$shape[which(triangle_data_train$ao == 0)][k],ncol = 1))
  
  set <- list(triangles = data, ao = label, shape = shapes)
  
  return(set)
}

#set up for generating a random sample of triangle shapes 
triangle_data_equilateral <- triangle_data_train$triangles[,,which(triangle_data_train$shape == 0)]
triangle_data_isosceles <- triangle_data_train$triangles[,,which(triangle_data_train$shape == 1)]
triangle_data_scalene <- triangle_data_train$triangles[,,which(triangle_data_train$shape == 2)]

#N must be divisible by 5
#Generate an N sample of equilateral isosceles and scalene triangles
sampling_eis <- function(N){
  k <- sample(min(dim(triangle_data_equilateral)[3],dim(triangle_data_isosceles)[3]))[1:(3*N/5)]
  equ_sample <- triangle_data_equilateral[,,k[1:(N/5)]]
  iso_sample <- triangle_data_isosceles[,,k[1:(N/5)]]
  sca_sample <- triangle_data_scalene[,,k]
  
  data <- abind(equ_sample,iso_sample)
  data <- abind(data,sca_sample)
  label <- rbind(matrix(triangle_data_train$ao[which(triangle_data_train$shape == 0)][k[1:(N/5)]],ncol = 1),
                  matrix(triangle_data_train$ao[which(triangle_data_train$shape == 1)][k[1:(N/5)]],ncol = 1),
                  matrix(triangle_data_train$ao[which(triangle_data_train$shape == 2)][k],ncol = 1))
  shapes <- rbind(matrix(rep(0,N/5),ncol = 1),matrix(rep(1,N/5),ncol = 1),matrix(rep(2,3*N/5),ncol = 1))
  
  set <- list(triangles = data, ao = label, shape = shapes)
  
  return(set)
}


#Samples for AO investigation
set.seed(10);n10 <- sampling_ao(10)
set.seed(50);n50 <- sampling_ao(50)
set.seed(100);n100 <- sampling_ao(100)
set.seed(500);n500 <- sampling_ao(500)
set.seed(1000);n1000 <- sampling_ao(1000)
set.seed(10000);n10000 <- sampling_ao(10000)

#Samples for AO investigation
set.seed(11);m10 <- sampling_eis(10)
set.seed(51);m50 <- sampling_eis(50)
set.seed(101);m100 <- sampling_eis(100)
set.seed(501);m500 <- sampling_eis(500)
set.seed(1001);m1000 <- sampling_eis(1000)
set.seed(10001);m10000 <- sampling_eis(10000)



#Riemannian distance from border for AO----
theta_border1 <- rbind(matrix(seq(asin(0.5)+0.0001,pi/2 -0.0001,length.out = 500),ncol = 1),matrix(seq(asin(0.5)+0.0001,pi/2 - 0.0001,length.out = 500),ncol = 1))
phi_border11 <- rbind(matrix(acos(0.5/sin(theta_border1[1:500])),ncol = 1),matrix(-acos(0.5/sin(theta_border1[501:1000])),ncol = 1))
phi_border22 <- rbind(matrix(acos(0.5/sin(theta_border1[1:500])),ncol = 1),matrix(-acos(0.5/sin(theta_border1[501:1000])),ncol = 1)) + 2*pi/3
phi_border33 <- rbind(matrix(acos(0.5/sin(theta_border1[1:500])),ncol = 1),matrix(-acos(0.5/sin(theta_border1[501:1000])),ncol = 1)) + 4*pi/3
b_points <- rbind(cbind(theta_border1,phi_border11),cbind(theta_border1,phi_border22),cbind(theta_border1,phi_border33))
border_points_ao <- array(c(1/sqrt(3),-1/sqrt(3),0,0,0,0),dim = c(3,2,3000))
for (i in 1:3000){
  border_points_ao[3,,i] <- matrix(c( sin(b_points[i,1])*sin(b_points[i,2]) / ( 1 + sin(b_points[i,1])*cos(b_points[i,2]) ) , 
                                   cos(b_points[i,1])/(1 + sin(b_points[i,1])*cos(b_points[i,2])) ),1,2)
}

#A function to determine the Riemannian distance between a shape and the class boundary
calc_dist_from_border_ao <- function(incorrectlabels) {
  triangles <- triangle_data_test$triangles[,,incorrectlabels]
  
  # Use mclapply() for parallel processing (adjust `mc.cores` based on your CPU)
  closest_distance <- mclapply(seq_along(incorrectlabels), function(j) {
    distances <- apply(border_points_ao, MARGIN = 3, function(border) {
      riemdist(triangles[,,j], border, reflect = TRUE)
    })
    min(distances)
  }, mc.cores = parallel::detectCores() - 1) # Use all but one core
  
  return(unlist(closest_distance))  # Convert list output to vector
}
calc_dist_from_border_ao(c(1,2,3))

#The code hashtagged out below may need to be ran in order to reset something in the computer for mclapply

# triangles <- triangle_data_test$triangles[,,c(1,2,3)]
# closest_distance <- lapply(seq_along(c(1,2,3)), function(j) {
#   distances <- apply(border_points_ao, MARGIN = 3, function(border) {
#     riemdist(triangles[,,j], border, reflect = TRUE)
#   })
#   min(distances)
# })
# closest_distance


#Riemannian distance from border for Shape----
iso_theta <- seq(0,pi/2-0.00001,length.out = 500)
iso_phi1 <- rep(0,500)
iso_phi2 <- rep(pi/3,500)
iso_phi3 <- rep(2*pi/3,500)
iso_phi4 <- rep(pi,500)
iso_phi5 <- rep(4*pi/3,500)
iso_phi6 <- rep(5*pi/3,500)

iso_border_points <- rbind(cbind(iso_theta,iso_phi1),cbind(iso_theta,iso_phi2),cbind(iso_theta,iso_phi3),
                           cbind(iso_theta,iso_phi4),cbind(iso_theta,iso_phi5),cbind(iso_theta,iso_phi6))

border_points_shape <- array(c(1/sqrt(3),-1/sqrt(3),0,0,0,0),dim = c(3,2,3000))
for (i in 1:3000){
  border_points_shape[3,,i] <- matrix(c( sin(iso_border_points[i,1])*sin(iso_border_points[i,2]) / ( 1 + sin(iso_border_points[i,1])*cos(iso_border_points[i,2]) ) , 
                                         cos(iso_border_points[i,1])/(1 + sin(iso_border_points[i,1])*cos(iso_border_points[i,2])) ),1,2)
}

#A function to determine the Riemannian distance between a shape and the class boundary
calc_dist_from_border_shape <- function(incorrectlabels) {
  triangles <- triangle_data_test$triangles[,,incorrectlabels]
  
  # Use mclapply() for parallel processing (adjust `mc.cores` based on your CPU)
  closest_distance <- mclapply(seq_along(incorrectlabels), function(j) {
    distances <- apply(border_points_shape, MARGIN = 3, function(border) {
      riemdist(triangles[,,j], border, reflect = TRUE)
    })
    min(distances)
  }, mc.cores = parallel::detectCores() - 1) # Use all but one core
  
  return(unlist(closest_distance))  # Convert list output to vector
}
calc_dist_from_border_shape(c(1,2,3))

#The code hashtagged out below may need to be ran in order to reset something in the computer for mclapply

# triangles <- triangle_data_test$triangles[,,c(1,2,3)]
# closest_distance <- lapply(seq_along(c(1,2,3)), function(j) {
#   distances <- apply(border_points_shape, MARGIN = 3, function(border) {
#     riemdist(triangles[,,j], border, reflect = TRUE)
#   })
#   min(distances)
# })
# closest_distance
# calc_dist_from_border_shape(c(1,2,3,4,5,6))

#####
#Bookstein data----
#bk.conversion turns the triangles in a triangle data set into a Nx2 matrix of bookstein coordinates 
bk.conversion <- function(data_set){
  bk <- apply(data_set$triangles,
                        MARGIN = 3, FUN = function(x) bookstein.shpv(x)) 
  bookstein_data <- cbind(matrix(bk[3,],ncol = 1),matrix(bk[6,],ncol = 1))
  bookstein_data[which(bookstein_data[,2] < 0),2] <- abs(bookstein_data[which(bookstein_data[,2] < 0),2])
  k <- sample(which(data_set$shape == 1),floor(length(which(data_set$shape == 1))/2))
  bookstein_data[k,1] <- -bookstein_data[k,1]
  set <- list(data = bookstein_data,ao = data_set$ao,shape = data_set$shape)
  
  return(set)
}
bookstein.test <- bk.conversion(data_set = triangle_data_test)

#Because of the separate metric on the hyperbolic plane Bokstein coordinates require their own distance functions
calc_dist_from_border_ao <- function(u.v){
  distances <- matrix(NA,300)
  x <- seq(-0.499,0.499,length.out = 100)
  border_points <- rbind(matrix(c(rep(-0.5,100),seq(0.1,100,length.out = 100)),100,2),
                         matrix(c(rep(0.5,100),seq(0.01,100,length.out = 100)),100,2),
                         matrix(c(x,sqrt(0.25 - x**2)),100,2))
  for (i in 1:300){
    A <- matrix(c(1,0,(border_points[i,1]-u.v[1])/u.v[2],border_points[i,2]/u.v[2]),2,2)
    D <- svd(A)$d
    distances[i] <- log(D[1]/D[2])
  }
  
  #set <- list(dist = min(distances), point = border_points[which(distances == min(distances)),])
  
  return(min(distances))
}
calc_dist_from_border_shape <- function(u.v){
  distances <- matrix(NA,300)
  x1 <- seq(-1.499,0.499,length.out = 100)
  x2 <- seq(-0.499,1.499,length.out = 100)
  border_points <- rbind(matrix(c(rep(-0.5,100),seq(0.1,100,length.out = 100)),100,2),
                         matrix(c(x1,sqrt(1 - (x1 + 0.5)**2)),100,2),
                         matrix(c(x2,sqrt(1 - (x2 - 0.5)**2)),100,2))
  for (i in 1:300){
    A <- matrix(c(1,0,(border_points[i,1]-u.v[1])/u.v[2],border_points[i,2]/u.v[2]),2,2)
    D <- svd(A)$d
    distances[i] <- log(D[1]/D[2])
  }
  
  #set <- list(dist = min(distances), point = border_points[which(distances == min(distances)),])
  
  return(min(distances))
}

#####
#Kendall data----
#kd.conversion turns the triangles in a triangle data set into a Nx2 matrix of Kendall spherical coordinates 
kd.conversion <- function(data_set){
  
  #Kendall coordinates for the upper plane
  kc <- apply(data_set$triangles,
              MARGIN = 3, FUN = function(x) (2/sqrt(3))*bookstein.shpv(x)) 
  kendall_data <- cbind(matrix(kc[3,],ncol = 1),matrix(kc[6,],ncol = 1))
  kendall_data[which(kendall_data[,2] < 0),2] <- abs(kendall_data[which(kendall_data[,2] < 0),2])
  k <- sample(which(data_set$shape == 1),floor(length(which(data_set$shape == 1))/2))
  kendall_data[k,1] <- -kendall_data[k,1]
  
  #3d Cartesian coordinates for plotting 
  rsq <- kendall_data[,1]**2 + kendall_data[,2]**2
  xk <- (1-rsq)/(2*(1+rsq))
  yk <- kendall_data[,1]/(1+rsq)
  zk <- kendall_data[,2]/(1+rsq)
  x.y.z <- cbind(xk,yk,zk)
  
  #Input data in form of (theta, phi)
  theta <- acos(2*zk)
  phi <- atan2(yk,xk) + pi
  theta.phi <- cbind(theta,phi)
  
  set <- list(polar = theta.phi,cart = x.y.z,plane = kendall_data,ao = data_set$ao,shape = data_set$shape)
  
  return(set)
}
kendall_input <- kd.conversion(data_set = triangle_data_train)
kendall.test <- kd.conversion(data_set = triangle_data_test)


####
#####
#The following code below is necessary for plotting graphs in the respective Kendall.Analysis and others, file
#Border plotting data for ao ---- 
theta_border <- rbind(matrix(seq(asin(0.5)+0.0001,pi/2,length.out = 1000),ncol = 1),matrix(seq(asin(0.5)+0.0001,pi/2,length.out = 1000),ncol = 1))
phi_border1 <- rbind(matrix(acos(0.5/sin(theta_border[1:1000])),ncol = 1),matrix(-acos(0.5/sin(theta_border[1001:2000])),ncol = 1))
phi_border2 <- rbind(matrix(acos(0.5/sin(theta_border[1:1000])),ncol = 1),matrix(-acos(0.5/sin(theta_border[1001:2000])),ncol = 1)) + 2*pi/3
phi_border3 <- rbind(matrix(acos(0.5/sin(theta_border[1:1000])),ncol = 1),matrix(-acos(0.5/sin(theta_border[1001:2000])),ncol = 1)) + 4*pi/3

x1 <- 0.5*sin(theta_border)*cos(phi_border1)
y1 <- 0.5*sin(theta_border)*sin(phi_border1)
z1 <- 0.5*cos(theta_border)

x2 <- 0.5*sin(theta_border)*cos(phi_border2)
y2 <- 0.5*sin(theta_border)*sin(phi_border2)
z2 <- 0.5*cos(theta_border)

x3 <- 0.5*sin(theta_border)*cos(phi_border3)
y3 <- 0.5*sin(theta_border)*sin(phi_border3)
z3 <- 0.5*cos(theta_border)

#Stereographic ao
X_border <- rbind(x1,x2,x3)/(1-rbind(z1,z2,z3))
Y_border <- rbind(y1,y2,y3)/(1-rbind(z1,z2,z3))

#Border plotting for shape----
iso_theta <- seq(0,pi/2,length.out = 500)
iso_phi1 <- rep(0,500)
iso_phi2 <- rep(pi/3,500)
iso_phi3 <- rep(2*pi/3,500)
iso_phi4 <- rep(pi,500)
iso_phi5 <- rep(4*pi/3,500)
iso_phi6 <- rep(5*pi/3,500)

xx1 <- 0.5*sin(iso_theta)*cos(iso_phi1)
yy1 <- 0.5*sin(iso_theta)*sin(iso_phi1)
zz1 <- 0.5*cos(iso_theta)

xx2 <- 0.5*sin(iso_theta)*cos(iso_phi2)
yy2 <- 0.5*sin(iso_theta)*sin(iso_phi2)
zz2 <- 0.5*cos(iso_theta)

xx3 <- 0.5*sin(iso_theta)*cos(iso_phi3)
yy3 <- 0.5*sin(iso_theta)*sin(iso_phi3)
zz3 <- 0.5*cos(iso_theta)

xx4 <- 0.5*sin(iso_theta)*cos(iso_phi4)
yy4 <- 0.5*sin(iso_theta)*sin(iso_phi4)
zz4 <- 0.5*cos(iso_theta)

xx5 <- 0.5*sin(iso_theta)*cos(iso_phi5)
yy5 <- 0.5*sin(iso_theta)*sin(iso_phi5)
zz5 <- 0.5*cos(iso_theta)

xx6 <- 0.5*sin(iso_theta)*cos(iso_phi6)
yy6 <- 0.5*sin(iso_theta)*sin(iso_phi6)
zz6 <- 0.5*cos(iso_theta)

XX_border <- rbind(xx1,xx2,xx3,xx4,xx5,xx6)/(1-rbind(zz1,zz2,zz3,zz4,zz5,zz6))
YY_border <- rbind(yy1,yy2,yy3,yy4,yy5,yy6)/(1-rbind(zz1,zz2,zz3,zz4,zz5,zz6))

#####
#Planar Data----
#planar.conversion turns the triangles in a triangle data set into a Nx2 matrix of planar coordinates 
planar.conversion <- function(data_set){
  tris <- data_set$triangles
  norm.side.lengths <- matrix(NA,dim(tris)[3],3)
  
  #calculating side lengths for all triangles 
  for (i in 1:dim(tris)[3]){
    tri <- tris[,,i]
    l1 <- norm(tri[1,] - tri[2,],type = '2')
    l2 <- norm(tri[3,] - tri[2,],type = '2')
    l3 <- norm(tri[1,] - tri[3,],type = '2')
    norm.side.lengths[i,] <- sort(c(l1,l2,l3)/max(c(l1,l2,l3)))
  }
  
  coords <- cbind(norm.side.lengths[,2],norm.side.lengths[,1])
  set <- list(planar = coords,ao = data_set$ao,shape = data_set$shape)
  
  return(set)
}
planar_input <- planar.conversion(data_set = triangle_data_train)
planar.test <- planar.conversion(data_set = triangle_data_test)

#####
#The following are necessary to be ran in order to generate graphs for the planar coordinates
#Border data for AO----
x <- seq(0,1,length.out = 10000)
y <- sqrt(1-x**2)
x.y <- cbind(x,y)

planar_dist_from_border_ao <- function(incorrectlabels){
  triangles <- planar.test$planar[incorrectlabels,]
  closest_distance <- c()
  for (i in 1:length(incorrectlabels)){
    distances_1 <- c()
    for (j in 1:dim(x.y)[1]){
      distances_1[j] <- norm(triangles[i,]-x.y[j,],type = '2')
    }
    closest_distance[i] <- min(distances_1)
  }
  return(closest_distance)
}
planar_dist_from_border_ao(c(1,3,4,5))


#Border data for Shape----
iso1x <- rep(1,1000)
iso1y <- seq(0,1,length.out = 1000)
iso2x <- seq(0.5,1,length.out = 1000)
iso2y <- seq(0.5,1,length.out = 1000)
border <- rbind(cbind(iso1x,iso1y),cbind(iso2x,iso2y))

planar_dist_from_border_shape <- function(incorrectlabels){
  triangles <- planar.test$planar[incorrectlabels,]
  closest_distance <- c()
  for (i in 1:length(incorrectlabels)){
    distances_1 <- c()
    for (j in 1:2000){
      distances_1[j] <- norm(triangles[i,]-border[j,],type = '2')
    }
    closest_distance[i] <- min(distances_1)
  }
  return(closest_distance)
}


#plotting data AO----
plot(planar.test$planar[which(planar.test$ao == 1),1],planar.test$planar[which(planar.test$ao == 1),2],col = 'red',pch = 20,asp = 1,xlim = c(0,1))
points(planar.test$planar[which(planar.test$ao == 0),1],planar.test$planar[which(planar.test$ao == 0),2],col = 'blue',pch = 20)
lines(x,sqrt(1-x**2),lwd = '2')
lines(x,x,lwd = '2')
lines(x,1-x,lwd = '2')
lines(rep(1,100),x,lwd = '2')

#####
#LQ decomposition----
#LQ.conversion turns the triangles in a triangle data set into a Nx2 matrix of bookstein coordinates 
LQ.conversion <- function(data_set){
  tris <- data_set$triangles
  
  #Calulate the lower triangular matrix 
  L <- array_reshape(apply(tris,MARGIN = 3,FUN = function(x) lq(defh(2)%*%x)$L), 
                     dim = c(2,2,dim(tris)[3]),order = 'F')
  
  #Store the Q's (may need later)
  Q <- array_reshape(apply(tris,MARGIN = 3,FUN = function(x) lq(defh(2)%*%x)$Q), 
                     dim = c(2,2,dim(tris)[3]),order = 'F')
  
  #Return the L's scaled by centroid size which is consistent with SSA
  # W <- array_reshape(apply(L,MARGIN = 3,FUN = function(x) x/centroid.size(x)),
  #                    dim = c(2,2,dim(tris)[3]),order = 'F')
  
  W <- array_reshape(apply(L,MARGIN = 3,FUN = function(x) x/norm(x,type = '2')),
                     dim = c(2,2,dim(tris)[3]),order = 'F')

  set <- list(LQ = W, ao = data_set$ao, shape = data_set$shape)
  
  return(set)
}
LQ_input <- LQ.conversion(data_set = triangle_data_train)
LQ.test <- LQ.conversion(data_set = triangle_data_test)

#####



