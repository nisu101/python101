#numpy experiments

import numpy as np
from StringIO import StringIO

'''
x = np.arange(5)      # creates an nd array with the elements 0,1,2,3,4
y = np.arange(5) * 2  # creates an nd array with the element  0,2,4,6,8
z = x + y
print(z)

x.dtype    # returns the implicit data type of the ndarray. The possible options are u(int16\32\64) float16\32\64.  
x.shape    # size and dimension of the ndarray
x.ndim     # of dimensions 
x.size     # size
x.itemsize # total bytes

'''
#using arange for making array with elements at specified gaps.
'''
x = np.arange(0,1,0.1) # elements starting from 0, ending at 1, at the gap of 0.1
print(x)
'''

# making a 3d array
'''
x = array([
  			[[0,1,2,3,4],[0,1,4,6,8]],
				[[3,4,5,6,7],[0,10,20,30,40]],
				[[0,10,40,60,80],[30,40,50,60,70]]])
print(x)
x.shape

'''
#accessing the elements
'''
print(x[0,0,0])
print(x[0,1,0])
print(x[1,1,4])
print(x[2,0,3])
'''

# using dtype to make a datatype - we can then make ndarrays of this data type
'''
t = dtype([('name',str_,50),('year',int32),('price',float32)])
t['name']

i = array([('Seinfeld',1992,60.43),('Two and  a half men',1996,34.25),('Grey''s Anatomy',1998,95)],dtype=t)
i[1]
'''

#indexing 1-d array
'''
x = np.arange(10) 
print(x)
print(x[2:5])	
print(x[:-1])
print(x([:7:2])
print(x[::-1])
x.shape = (2,5)
print(x) # reshaping 10x1 to 2x5 
'''
'''
'''

#indexing multi-dimensional array
'''
x = arange(24).reshape(2,3,4)
print(x[:,0,0])

print(x[0,:,:])
print(x[0, ...]) # same as above
print(x[0,1,::2]) # slicing and indexing can be done simaltaneously
print(x[0,:,-1] # negative indexing also works
print(x[0,::2,-1] # -ve indexing works even with slicing
'''
#Some shape manipulating functions
'''
x = arange(24).reshape(2,3,4)
y = x.ravel()
y = x.flatten() # same as ravel, but the only diff is flatten always creates new memory to store stuff.
x.shape = (12,2) # you can change the shape directly, as long as the new element has the same #of elements
x.transpose() # transpose
x.resize(12,2) # same as reshape, but resize modifies the array it operates on
'''

#stacking functions
'''
x = arange(9).reshape(3,3)
y = 3*x

hstack(x,y) # Horizontal Stacking
concatenate((a,b), axis = 1) # this also does horizontal stacking
vstack(x,y) # Vertical Stacking
concatenate((a,b), axis = 0) # this also does vertical stacking
dstack(x,y) # Depth Stacking
'''	
# Splitting functions
'''
x = arange(9).reshape(3,3)	
hsplit(x,3) #horizontal split
split(x,3,axis=0) # This does the same thing as above
vsplit(x,3) # Vertical split
split(x,3,axis=1) # same as above
dsplit(x,3)		
'''

# Conversion to list
'''
x = arange(9).reshape(3,3)	
x.tolist()
'''

# using loadtxt to import from CSVs
'''
c,v = loadtxt('/home/nisarg/tech/pythonProgramming/images/data.csv',delimiter=',',unpack=True)
'''
# Some statistical functions
'''
ave  = numpy.average(c)
vawp = average(c,weights = v)
cmax = numpy.max(c)
cmin = numpy.min(c)
cmed = numpy.median(c)
cmod = numpy.mode(c)
cvar = numpy.var(c)
clog = numpy.log(c)
cdif = numpy.diff(c)
cstd = numpy.std(c)
vals = c[where(c < 10000)]
amax = argmax(c)
amin = argmin(c)
x = array([1,2,3,4,6])
print(x.prod())
print(x.cumprod())
'''
# fill and clip - data arranging/processing functions
'''
p    = arange(10)
p.fill(1501)
ccl  = clip(c,1000,10000) #clips values <1000 to 1000, and >10000 to 10000
print(ccl)
c.compress(c>1000)
'''
# covariance, co-efficients etc.
'''
X = array([1,1.5,1.7,2.1,2.2,2.4,2.7,3,3.1,3.3]);
Y = array([.6,.62,.67,.52,.71,.74,.71,.54,.67,.88]);
c = cov(X,Y)
c.diagonal()
c.trace()
cc = corrcoef(X,Y)
'''
# unique
'''
C = [1,1,1,1,1,1,2,2,2,2,3,3,4,5,6]
uc = unique(C)
'''
# Least squares
'''
(x, residuals, rank, s) = numpy.linalg.lstsq(A, b)
'''
#vectorize for enable a function for element-by-element type of processing on numpy arrays (this is like map for numpy arrays)
'''
def f(n,otype=np.float):
	if(n > 0.3):
		return n
	return 0.1

v = np.array([0.1, 0.1, 0.3, 0.1, 0.4, 0.32, 0.24, 0.65, 0.66, 0.07, 0.87])
f2 = np.vectorize(f)

v2 = f2(v)
print(v2)
'''

#Matrix manipulation
'''
M = np.mat('1 2 3; 4 5 6; 7 8 9') #Creating matrix from a string
M = np.mat(array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3,3)) #Creating matrix from an ndarray
M.T #Transpose
M.I #Inverse 
A = np.eye(3) #identity
B = A * 2
C = np.bmat("A B; A B; A B") # Makes a compound matrix
C = add(A,B) # use of universal funcs - add is one such function
'''

#Linear Algebra using linalg
'''
M  = np.mat('0 1 2; 1 0 3; 4 -3 8 ') #Creating matrix from a string
Mi = np.linalg.inv(M)  # inverse using linalg

A = np.mat('1 -2 1;	0 2 -8; -4 5 9')
b = np.array([0, 8,-9])

x = np.linalg.solve(A,b) # This solves Ax = b
np.dot(A,x) # dot us for dot products

A = np.mat([[3, -2], [1, 0]])
[evals,evecs] = np.linalg.eig(A) # eigenvalues and eigenvectors

A = np.mat("4 11 14;8 7 -2")
[U,S,V] = np.linalg.svd(A) #SVD

A = np.mat("4 11 14;8 7 -2")
p = np.linalg.pinv(A) # pseudo-inverse

A = np.mat([[1, 2],[3,4]])
d = np.linalg.det(A) # determinant
'''
#Random number generation
'''
nums = np.random.normal(size=100)
'''
# Making custom ufuncs 
'''
def ultimate_func(a):
	#r = np.zeros_like(a)
	#r.flat = 39
	r = 39
	return r

def modulo10(a):
	r = a%10
	return r

uf = np.frompyfunc(ultimate_func,1,1)
A = uf(np.arange(16).reshape(4,4))
print(A)	

uf = np.frompyfunc(modulo10,1,1)
A = uf(np.arange(16).reshape(4,4))
print(A)	
'''
# Specific ufunc methods - Reduce
'''
a = np.array([[1,2,21,34,3,45],[11,21,211,341,31,451]])
print(np.add.reduce(a,1)) #The second argument to do the row-by-row sum, instead of colum-by-column
'''

# Specific ufunc methods - Reduceat
'''
a = np.arange(9)
print(np.add.reduceat(a,[0,5,2,7])) #Reduce of a[0:5],a[5],a[2:7] and 7:end
'''

# Specific ufunc methods - accumulate
'''
a = np.array([[1,2,21,34,3,45],[11,21,211,341,31,451]])
print(np.add.accumulate(a,1)) #The second argument to do the row-by-row sum, instead of colum-by-column
'''
# Specific ufunc methods - outer
'''
a = np.array([1,2,21,34,3,45])
b = np.array([1,2,3,4,5])
print(np.add.outer(a,b))
print(np.multiply.outer(a,b))
'''

#sorting
'''
A = np.array([0, 34, 3, 23, 4, 22, 54, 6, 4325, 3, 1245])
B = np.sort(A) # plain and simple sorting
I = np.argsort(A) # returns the indices that would sort an array
'''
#broadcasting
'''
x = np.arange(35).reshape(5,7)
#print(x)

y = x[np.array([0,2,4]),np.array([1,2,3])]
print(y)
'''

#using linspace
'''
x = np.linspace(0,1,8) # 8 elements starting from 0, ending at 1, at regular interval
print(x)
'''
#using genfromtxt
'''
data = "1 2 3\n 4 5 6\n 7 8 9";
x = np.genfromtxt(StringIO(data),delimiter=" ")
print(x)
x = np.genfromtxt(StringIO(data),delimiter=" ",usecols = (0,-1))
print(x)
'''

#extract (can provide operations similar to logical indexing)
'''
A = np.array([0, 34, 3, 23, 4, 22, 54, 6, 4325, 3, 1245])
condition = A > 20

B = extract(condition,A)
'''

#masking (can provide operations similar to logical indexing)
'''
x = np.arange(35).reshape(5,7)
y = x>20
print(x[y])
'''
