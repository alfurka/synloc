import synloc as s

data3d = s.sample_trivariate_xyz(500)
assert data3d.shape[0] > 0
assert data3d.shape[1] == 3

data2d = s.sample_circulars_xy(500)
assert data2d.shape[0] > 0
assert data2d.shape[1] == 2

syn1 = s.clusterCov(data2d , n_clusters=30, size_min=15)
syn2 = s.clusterCov(data3d , n_clusters=30, size_min=15, size_max = 40)

syn1.fit()
assert syn1.synthetic.shape == data2d.shape
syn2.fit()
assert syn2.synthetic.shape == data3d.shape

syn3 = s.clusterGaussCopula(data2d , n_clusters=30, size_max = 40)
syn4 = s.clusterGaussCopula(data3d , n_clusters=30, size_min=15, size_max = 40)

syn3.fit()
assert syn3.synthetic.shape == data2d.shape
syn4.fit()
assert syn4.synthetic.shape == data3d.shape

syn1.comparePlots(syn1.data.columns.values[0])
syn2.comparePlots(syn2.data.columns.values[0])
syn3.comparePlots(syn3.data.columns.values[0])
syn4.comparePlots(syn4.data.columns.values[0])

syn1.comparePlots(syn1.data.columns.values)
syn2.comparePlots(syn2.data.columns.values)
syn3.comparePlots(syn3.data.columns.values)
syn4.comparePlots(syn4.data.columns.values)


