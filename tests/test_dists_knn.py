import synloc as s

data3d = s.sample_trivariate_xyz(1000)
assert data3d.shape[0] > 0
assert data3d.shape[1] == 3

data2d = s.sample_circulars_xy(1000)
assert data2d.shape[0] > 0
assert data2d.shape[1] == 2

syn1 = s.LocalCov(data2d, K = 20)
syn2 = s.LocalCov(data3d, K = 20)

syn1.fit()
assert syn1.synthetic.shape == data2d.shape
syn2.fit()
assert syn2.synthetic.shape == data3d.shape


syn1.comparePlots(syn1.data.columns.values[0])
syn2.comparePlots(syn2.data.columns.values[0])

syn1.comparePlots(syn1.data.columns.values)
syn2.comparePlots(syn2.data.columns.values)



