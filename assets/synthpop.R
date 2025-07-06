# Install required packages if not already installed
# install.packages(
#  c("parallelly","future","future.apply","Rsolnp","synthpop"),
#  type = "binary"
# )


# Load necessary libraries
library(synthpop)

set.seed(42)  # Set seed for reproducibility

# Read input datasets
df1 = read.csv('california.csv')
df2 = read.csv('circulars.csv')
df3 = read.csv('trivariate.csv')


# Generate synthetic data for each dataset using synthpop
s1 = syn(df1, proper = TRUE, seed = 1, cart.minbucket = 10, smoothing = 'spline')
s2 = syn(df2, proper = TRUE, seed = 2, cart.minbucket = 10, smoothing = 'spline')
s3 = syn(df3, proper = TRUE, seed = 3, cart.minbucket = 10, smoothing = 'spline')


# Save synthetic datasets to CSV files
write.csv(s1$syn, 'california_synthpop.csv', row.names = FALSE)
write.csv(s2$syn, 'circulars_synthpop.csv', row.names = FALSE)
write.csv(s3$syn, 'trivariate_synthpop.csv', row.names = FALSE)

