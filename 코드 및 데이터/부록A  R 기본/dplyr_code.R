library(dplyr)

data(iris)

iris %>% ### Select rows (Species == 'setosa')
  filter(Species == 'setosa')

iris %>% ### Select columns (Species == 'setosa')
  select(Sepal.Length, Species)

iris %>% ### Generate new variable
  mutate(leng.per.wid =  Sepal.Length/Sepal.Width)
