library(shapes)

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

equilateral_tri <- function(){
  H <- defh(2)
  new_tri <- rand_affine(H)
  return(new_tri)
}

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

scalene_tri <- function(){
  tri <- matrix(rnorm(6),2,3)
  new_tri <- rand_affine(tri)
  return(new_tri)
}

ao.identifier <- function(X){
  a <- norm(X[,1] - X[,2],type = '2')**2
  b <- norm(X[,1] - X[,3],type = '2')**2 
  c <- norm(X[,3] - X[,2],type = '2')**2 
  
  if (max(a,b,c) < (a + b + c - max(a,b,c))){return(1)}
  
  else{return(0)}
}

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
      shape.label[i] <- 1
    }
    
    if (j == 2){
      tri <- isos_tri()
      data[,,i] <- t(tri)
      ao.label[i] <- ao.identifier(tri)
      shape.label[i] <- 2
    }
    
    if (j %in% 3:6){
      tri <- scalene_tri()
      data[,,i] <- t(tri)
      ao.label[i] <- ao.identifier(tri)
      shape.label[i] <- 3
    }
  }
  
  set <- list(triangles = data,ao = ao.label,shape = shape.label)
  
  return(set)
}

triangle_data <- triangle_function(50000)

#Bookstein data----
#bk.conversion turns the triangles in triangle_data into a Nx2 matrix of booksteing coordinates 
bk.conversion <- function(data_set){
  bk <- apply(data_set$triangles,
                        MARGIN = 3, FUN = function(x) bookstein.shpv(x)) 
  bookstein_data <- cbind(matrix(bk[3,],ncol = 1),matrix(bk[6,],ncol = 1))
  bookstein_data[which(bookstein_data[,2] < 0),2] <- abs(bookstein_data[which(bookstein_data[,2] < 0),2])
  k <- sample(which(data_set$shape == 2),floor(length(it)/2))
  bookstein_data[k,1] <- -bookstein_data[k,1]
  set <- list(data = bookstein_data,ao = data_set$ao,shape = data_set$shape)
  
  return(set)
}
bookstein_input <- bk.conversion(data_set = triangle_data) 
#####

#Kendall data----
kd.conversion <- function(data_set){
  
  #Kendall coordinates for the upper plane
  kc <- apply(data_set$triangles,
              MARGIN = 3, FUN = function(x) (2/sqrt(3))*bookstein.shpv(x)) 
  kendall_data <- cbind(matrix(kc[3,],ncol = 1),matrix(kc[6,],ncol = 1))
  kendall_data[which(kendall_data[,2] < 0),2] <- abs(kendall_data[which(kendall_data[,2] < 0),2])
  k <- sample(which(data_set$shape == 2),floor(length(which(data_set$shape == 2))/2))
  kendall_data[k,1] <- -kendall_data[k,1]
  
  #3d Cartesian coordinates for plotting 
  rsq <- kendall_data[,1]**2 + kendall_data[,2]**2
  xk <- (1-rsq)/(2*(1+rsq))
  yk <- kendall_data[,1]/(1+rsq)
  zk <- kendall_data[,2]/(1+rsq)
  x.y.z <- cbind(xk,yk,zk)
  
  #Input data in form of (theta, phi)
  theta <- acos(2*zk)
  phi <- atan2(yk,xk)
  theta.phi <- cbind(theta,phi)
  
  set <- list(polar = theta.phi,cart = x.y.z,plane = kendall_data,ao = data_set$ao,shape = data_set$shape)
  
  return(set)
}
kendall_input <- kd.conversion(data_set = triangle_data)
#plotting data----
x <- kendall_input$cart[which(kendall_input$ao == 0),1]
y <- kendall_input$cart[which(kendall_input$ao == 0),2]
z <- kendall_input$cart[which(kendall_input$ao == 0),3]
x1 <- kendall_input$cart[which(kendall_input$shape == 2),1]
y1 <- kendall_input$cart[which(kendall_input$shape == 2),2]
z1 <- kendall_input$cart[which(kendall_input$shape == 2),3]
theta <- kendall_input$polar[,1]
phi <- kendall_input$polar[,2]
scatter3Drgl(x,y,z,col = 'red',pch = 20)
scatter3Drgl(x1,y1,z1,col = 'red',pch = 20)
scatter3D(0,0,0.5,col = 'black',add = TRUE)
plotrgl(smooth = TRUE)
scatter3Drgl(0.5*sin(theta)*cos(phi),0.5*sin(theta)*sin(phi),0.5*cos(theta),aspect3d(x = 1,y = 1,z = 1))
plotrgl(smooth = TRUE)

plot(kendall_input$plane[which(kendall_input$shape == 2),],xlim = c(-2,2),ylim = c(0,2),asp = 1)
points(kendall_input$plane[which(kendall_input$shape == 1),],col = 'red',pch = 20)
u <- sin(theta)*sin(phi)/(1 + sin(theta)*cos(phi))
v <- cos(theta)/(1 + sin(theta)*cos(phi))
plot(cbind(u,v)[which(kendall_input$shape == 2),],xlim = c(-2,2),ylim = c(0,2),asp = 1)



plot(kendall_data,pch = 20,xlim = c(-2,2),ylim = c(0,2),asp = 1)

abline(v = 1/sqrt(3))
abline(v = -1/sqrt(3))
abline(h = 0)
abline(h = 1)
sum(data_set$ao)








#####

#Planar Data----
plane.conversion <- function(data_set){
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
plane_input <- plane.conversion(data_set = triangle_data)
#plotting data----

plot(planar_input$planar[which(planar_input$shape == 3),],xlim = c(0,1),ylim = c(0,1),pch = 20,asp = 1,col = 'green')
points(planar_input$planar[which(planar_input$shape == 2),],col = 'red',pch = 20)
points(planar_input$planar[which(planar_input$shape == 1),],col = 'blue',pch = 20)
x <- seq(0,1,by = 0.1)
lines(x,x)
lines(x,1-x)


#####

#QR decomposition----
library(tensr)
LQ_conversion <- function(data_set){
  tris <- data_set$triangles
  
  #Calulate the lower triangular matrix 
  L <- array_reshape(apply(tris,MARGIN = 3,FUN = function(x) lq(defh(2)%*%x)$L), 
                     dim = c(2,2,dim(tris)[3]),order = 'F')
  
  #Store the Q's (may need later)
  Q <- array_reshape(apply(tris,MARGIN = 3,FUN = function(x) lq(defh(2)%*%x)$Q), 
                     dim = c(2,2,dim(tris)[3]),order = 'F')
  
  #Return the L's scaled by centroid size which is consistent with SSA
  W <- array_reshape(apply(L,MARGIN = 3,FUN = function(x) x/centroid.size(x)),
                     dim = c(2,2,dim(tris)[3]),order = 'F')
  

  set <- list(LQ = W, ao = data_set$ao, shape = data_set$shape)
  
  return(set)
}
LQ_input <- LQ_conversion(data_set = triangle_data)
#----
#attempt at step by step riemannanian distance 
x1 <- triangle_data$triangles[,,1]
Hx1 <- defh(2)%*%x1
L <- lq(Hx1)$L
w1 <- L/centroid.size(L)
acos(w1[1,1])
riemdist(x1,mu)
LQ_input$LQ[,,1]

acos(1/sqrt(1 + kendall_input$plane[1,1]**2 + kendall_input$plane[1,2]**2))



x1 <- triangle_data$triangles[,,1]
x2 <- triangle_data$triangles[,,2]
z1 <- (defh(2)%*%x1)/centroid.size(defh(2)%*%x1)
z2 <- (defh(2)%*%x2)/centroid.size(defh(2)%*%x2)
sum(svd(t(z2)%*%z1)$d)
riemdist(x1,x2)













