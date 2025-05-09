##### R basic 5 #####

## ----quadratic_plot-----------------------------------------------
x <- seq(from = -1, to = 1, by = 0.1)
y <- x^2
plot(x, y)


## ----abline_ab, -----------------------------------------
x = seq(from = 0, to = 1, by = 0.1)
y = x^2
plot(y ~ x)
abline(a = 0, b = 1)   #intercept를 0으로, slope을 1로 설정


## ----quadratic_plot_dense------------------------------------------------
x <- seq(from = -1, to = 1, by = 0.01)
y <- x^2
plot(x, y)

## ----plot_type_line------------------------------------------------------
plot(x, y, type = "l")

## ----exercise_1----------------------------------------------------------
x <- seq(from = -2, to = 2, by = 0.01)
y <- x^3
plot(x, y, type = 'l')

## ----exercise_2----------------------------------------------------------
x <- seq(from = 0.5, to = 1.5, by = 0.01)
y <- log(x)
plot(x, y, type = 'l')

## ----exercise_3----------------------------------------------------------
x <- seq(from = 0, to = 6 * pi, by = 0.01)
y <- sin(x)
plot(x, y, type = 'l')

## ----points--------------------------------------------------------------
x <- seq(from = 0, to = 1, by = 0.01)
y1 <- x^2
y2 <- x
plot(x, y1, col = 'blue', type = 'l')
lines(x, y2, col = 'red')

## ----lines_fail----------------------------------------------------------
x <- seq(from = 0, to = 1, by = 0.01)
y1 <- x^2
y2 <- x + 0.75
plot(x, y1, col = 'blue', type = 'l')
lines(x, y2, col = 'red')

## ----cbind_matplot-------------------------------------------------------
y <- cbind(y1, y2)

matplot(x, y, type = 'l')
?matplot

## ----matplot_options-----------------------------------------------------
y <- cbind(y1, y2)
matplot(x, y, type = 'l', col = c("red", "blue"), lty = 1)

## ----exercise_4----------------------------------------------------------
x <- seq(from = 0, to = 2 * pi, by = 0.01)

# sin(x) and cos(x): input as *radians* rather than degrees
y1 <- sin(x)
y2 <- cos(x)
y3 <- 2 * sin(x + pi/4)
y <- cbind(y1, y2, y3)
matplot(x, y, type = 'l', col = c("black", "red", "blue"), lty = 1)

