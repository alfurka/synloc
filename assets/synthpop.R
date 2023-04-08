library(synthpop)
library(ggplot2)
library(patchwork)

df1 = read.csv('california.csv')
df2 = read.csv('circulars.csv')
df3 = read.csv('trivariate.csv')


s1 = syn(df1, proper = TRUE, seed = 1, cart.minbucket = 10, smoothing = 'spline')

s2 = syn(df2, proper = TRUE, seed = 2, cart.minbucket = 10, smoothing = 'spline')

p1 = ggplot(df2, aes(x=x, y=y)) + geom_point()
p2 = ggplot(s2$syn, aes(x=x, y=y)) + geom_point()
p1+p2

s3 = syn(df3, proper = TRUE, seed = 3, cart.minbucket = 10, smoothing = 'spline')


p1 = ggplot(df3, aes(x=z, y=y)) + geom_point()
p2 = ggplot(s3$syn, aes(x=z, y=y)) + geom_point()
p1+p2


write.csv(s1$syn, 'california_synthpop.csv', row.names = FALSE)
write.csv(s2$syn, 'circulars_synthpop.csv', row.names = FALSE)
write.csv(s3$syn, 'trivariate_synthpop.csv', row.names = FALSE)

