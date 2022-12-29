library(synthpop)

df1 = read.csv('california.csv')
df2 = read.csv('circulars.csv')
df3 = read.csv('trivariate.csv')


s1 = syn(df1, proper = TRUE, seed = 1)
write.csv(s1$syn, 'california_synthpop.csv', row.names = FALSE)

s2 = syn(df2, proper = TRUE, seed = 2)
write.csv(s2$syn, 'circulars_synthpop.csv', row.names = FALSE)

s3 = syn(df3, proper = TRUE, seed = 3)
write.csv(s3$syn, 'trivariate_synthpop.csv', row.names = FALSE)