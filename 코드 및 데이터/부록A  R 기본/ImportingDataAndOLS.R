##### Importing data and OLS estimation #####

# change directory and import data
getwd()
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
dir()


## Load data ?پ??? ?????? ??????  

# Importing a csv file
data1 <- read.csv("DataHousingPrice.csv")

# import txt file
data2 = read.table("DataHousingPrice.txt",header=T)


#### import excel data
#install.packages("readxl")
library("readxl")

# import data from an xls file
dataxls = read_excel("DataHousingPrice.xls")

# import data in the first sheet in an xlsx file
dataxlsx = read_excel("DataHousingPrice.xlsx",sheet=1) 


# Exercise
# import data in the second sheet in a xlsx file


# Regress housing price on number of bedrooms
ols = lm(PRICE~BDRMS,data=data1)
summary(ols)


attach(data1)   
Y <- cbind(PRICE)
X <- cbind(LOTSIZE, SQRFT, BDRMS)

# Orternatively, matrix?? ???? 
data = as.matrix(read.csv("DataHousingPrice.csv",header=T))

y = cbind(data[,1])   # or y = matrix(data[,1])
x = cbind(data[,2:4])
head(x)


## create new dataset without missing data if there are missing data

#Case1: no missing data

# NA?? ?ִ????? Ȯ??
complete.cases(data)
n    = nrow(data)
n

# Case2: there are missing data
dataMissing = read.csv("DataHousingPriceMISSING.csv",header=T)

complete.cases(dataMissing)
# na.omit?? NA(missing data)?? ��?? 
dataCleaned = na.omit(dataMissing)
nCleaned    = nrow(dataCleaned)
nCleaned



# write data
write.csv(dataCleaned,"dataCleaned.csv")
write.table(dataCleaned,"dataCleaned.txt")

#install.packages("writexl")
library(writexl)

write_xlsx(dataCleaned, "dataCleaned.xlsx")


#################
## OLS estimation
# Regress housing price on number of bedrooms

PRICE = data[,1]
BDRMS = data[,4]

fit0 = lm(PRICE~BDRMS)
summary(fit0)

plot(PRICE ~ BDRMS)
abline(fit0)
# alternatively
abline(lm(PRICE~BDRMS))

## ----abline?? sample mean ?߰? ---------------------------------
plot(PRICE ~ BDRMS)
abline(fit0)
abline(v = mean(BDRMS, na.rm = TRUE),
       h = mean(PRICE, na.rm = TRUE),
       col = 'red', lty = 2)


# Regress PRICE on SQRFT and BDRMS
LOTSIZE = data[,2]
SQRFT = data[,3]

Y = PRICE
X = cbind(SQRFT, BDRMS)

fit1 = lm(Y ~ X)
summary(fit1)


# Exercise
# Regress log(PRICE) on log(LOTSIZE), log(SQRFT) and Bedrooms


####################################
# OLS estimation using matrix
####################################
n    = nrow(data)
y = matrix(data[,1])
x = cbind(1,data[,2:4])
head(x)

invx = solve(t(x)%*%x)
olsb = invx%*%t(x)%*%y
olsb

#fit2 = lm(y~x-1) # regression without intercept
#summary(fit2)

