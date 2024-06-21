import numpy
from scipy.io import savemat

arr1 = numpy.array([1, 2, 3, 4, 5])
arr2 = numpy.array([1])

my_dict = {'arr1': arr1, 'arr2': arr2}
savemat('my_dict.mat', my_dict)

#get the data from mat
from scipy.io import loadmat
data = loadmat('my_dict.mat')
print(data['arr1'])

kk = []
test = loadmat('features/feature_1.mat')
kk.append(test)
print(test['feature'])
print(kk[0])