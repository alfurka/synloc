import synloc as s

data3d = s.sample_trivariate_xyz(1000)
assert data3d.shape[0] > 0
assert data3d.shape[1] == 3

data2d = s.sample_circulars_xy(1000)
assert data2d.shape[0] > 0
assert data2d.shape[1] == 2

syn1 = s.clusterCov(data2d , n_clusters=30, size_min=8)
syn2 = s.clusterCov(data3d , n_clusters=30, size_min=8)

syn1.fit()
assert syn1.synthetic.shape == data2d.shape
syn2.fit()
assert syn2.synthetic.shape == data3d.shape


syn1.comparePlots(syn1.data.columns.values[0])
syn2.comparePlots(syn2.data.columns.values[0])


syn1.comparePlots(syn1.data.columns.values)
syn2.comparePlots(syn2.data.columns.values)


print("original data distances for 2D:")
print(syn1.data_distances.mean(), "±", syn1.data_distances.std())
print("synthetic data distances for 2D:")
print(syn1.synthetic_distances.mean(), "±", syn1.synthetic_distances.std())

print("original data distances for 3D:")
print(syn2.data_distances.mean(), "±", syn2.data_distances.std())
print("synthetic data distances for 3D:")
print(syn2.synthetic_distances.mean(), "±", syn2.synthetic_distances.std())
