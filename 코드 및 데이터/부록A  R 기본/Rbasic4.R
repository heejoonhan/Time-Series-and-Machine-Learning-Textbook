##### R basic 4 #####

## matrix 
y0=seq(1,9)

?matrix
y=matrix(y0,nrow=3,ncol=3,byrow=FALSE)   
y
y=matrix(y0,nrow=3,ncol=3,byrow=TRUE)
y
y[2,1]
y[,1]
y[,c(2,3)]
y[c(1,2),]
y[c(1,3),]
y[,c(1,3)]

mat1 = matrix(seq(1:10),nrow=5, ncol=2,byrow=FALSE)
mat1

mat2= matrix(seq(1:10),nrow=2, ncol=5,byrow=FALSE)
mat2

mat3= matrix(seq(1:10),nrow=2, ncol=5,byrow=TRUE)
mat3 

# matrix multiplication : %*%, 행렬의 차원 주의 
mat1%*%mat2

# entry-wise(element by element) multiplication : *
mat1*mat2
mat2*mat3

# entry-wise(element by element) division : /
mat2/mat3

# transpose : t(A) 전치행렬 
t(mat1)

# diag(k) : k x k identity matrix
diag(3)

# diag(m) : 대각원소 값이 m인 대각행렬(diagonal matrix)
m=rep(1:3, times=2)
diag(m)

# det(x) : determinant(행렬식) 계산
det(diag(m))

# inverse : solve(A) if A is square matrix
A = matrix(c(4, 7, 2, 6), nrow = 2, byrow = TRUE)
A
A_inv = solve(A)
A_inv

# A'B : crossprod(A,B)
crossprod(mat2,mat3)
t(mat2)%*%mat3

crossprod(mat2)
t(mat2)%*%mat2

# AB'
mat2%*%t(mat3)


## random number generation 랜덤 넘버 만들기

#rnorm(n) generates n normal random numbers from the standard normal distribution.
d=rnorm(1) 
d1=rnorm(1, mean=10, sd=1)

#rnorm(n, a, b)=rnorm(n, mean=a, sd=b) : normal distribution with mean a and standard deviation b.

#runif(n) generates n uniform random numbers between 0 and 1. Uniform distribution [0,1]
d2=runif(1)

#runif(n, a, b) generates n uniform random numbers between a and b. Uniform distribution [a,b]
d3 = runif(1, 5, 10)


#####################
## if, else, else if  
#####################

d=rnorm(1) ; d
# if else (조건문 예시)
if (d<0.5){
  print("Less than 0.5")
} else if (d<0.75){
  print("Less than 0.75") 
} else{
  print("greater than 0.75")
}

print ("true d"); print(d);

# 조건문 예시 
x = runif(1) + 0.5; x
if (x >= -0.5 & x <= 0.5){
  if (x<0) {print(x); print("x is negative"); print(abs(x))
  }  else {print(x); print("x is positive")}
}else {print("x is wrong number")
}







#####################
##  loop(반복문)
#####################

for (i in 1:4) print(i)

for (i in 1:4) {
  print(i)
}

isum = 0
for (i in 100:200) {
  isum = isum + i
}
print(isum)

# 문자 변수의 looping
transport = c("bus", "subway", "car", "bike")
for (vehicle in transport) {
  print(vehicle)
}

#100을 넘으면서 100과 가장 가까운 t 구하기
# while 이용 
t = 0
tsum = 0
while (tsum <= 100) {
  t = t + 1
  tsum = tsum + t
}
print(t) ; print(tsum)  

# repeat 이용 
t = 0
tsum = 0
repeat{
  t = t + 1
  tsum = tsum + t
  if(tsum > 100) break
}
print(t) ; print(tsum)  


#####

n=500; #sample size
rho=0.4 #AR coefficient
x=matrix(0,n,1) #zero vector, dimension n*1 
head(x); tail(x)

eps = matrix(rnorm(n),n,1) # generate random vector from standard normal dist, eps is the error term
head(eps)
length(eps)


# x is an AR(1) process.
for (i in 2:n) {
  x[i,1] = rho*x[i-1,1]+eps[i,1]
}
head(x)
tail(x) #cat(x)


# first differencing dx_t=x_(t) - x_(t-1)
dx1=matrix(0,n-1,1)
for (i in 2:n){
  dx1[i-1,1]=x[i]-x[i-1]
}
head(dx1)
tail(dx1)


dx2=diff(x) # first difference
cbind(dx1,dx2)  # they are same


## OLS estimation

#first, we generate x and y
n=1000; 
b0=0.1; b1=1 #true parameter

y=matrix(0,n,1)
x=matrix(rnorm(n),n,1)
eps = matrix(rnorm(n),n,1)

for (i in 1:n){
  y[i]=b0+b1*x[i]+eps[i]
}

plot(x,y)
plot(y~x)

# Estimate linear model with R default function
fit1 = lm(y~x)
summary(fit1)
?lm

# 95% confidence interval
confint(fit1, level=0.95)

# Plotting regression line
abline(fit1)


yhat = fitted(fit1) # fitted value of y 
summary(yhat)
plot(yhat~x, type = 'l')

uhat = resid(fit1)  # residuals
summary(uhat)
plot(uhat~x)



####################################
# OLS estimation using matrix
####################################

x=cbind(1,x) #the first column in x corresponds to intercept
?cbind

head(x)
invx = solve(crossprod(x))    # (x'x)^(-1)
olsb = invx%*%crossprod(x,y)  # (x'x)^(-1)x'y
invx = solve(t(x)%*%x)
olsb  #OLS estimate, compare this with fit1

#fit2 = lm(y~x-1) # regression without intercept
#summary(fit2)

u    = y-x%*%olsb   #residual
sig2 = crossprod(u)/(n-2)
varcov = matrix(sig2,2,2) * invx
varcov
se = rbind(sqrt(varcov[1,1]), sqrt(varcov[2,2]))  #standard error
tvalue = olsb/se # t-statistic
cbind(olsb, se, tvalue)
