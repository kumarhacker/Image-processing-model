
import pandas as pd
import matplotlib.pyplot as plt

def cost_fun(X, Y, b, m):
    totalError = 0
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(X))

def gradient_descent(X, Y, b, m, learning_rate , iter_num):
	N = float(len(X))
	for i in range(iter_num):
		b_grad = 0
		m_grad = 0	
		for i in range(0, len(X)):
			x=X[i]
			y=Y[i]
			b_grad += -(2/N) * (y - ((m * x) + b))
			m_grad += -(2/N) * x * (y - ((m * x) + b))
		b = b - (learning_rate * b_grad)
		m = m - (learning_rate * m_grad)
	new_b = b
	new_m = m   
	return [new_b, new_m]


def main():
	data = pd.read_csv("train.csv").as_matrix()
	learning_rate = 0.0001
	inercept = 0 # initial y-intercept guess
	slope = 0 # initial slope guess
	iter_num = 1000
	x=data[:,0]
	y=data[:,1]

	print ("Intial cost: ",cost_fun(x, y, inercept, slope))

	plt.scatter(x, y, marker = 'x', color='g')

	plt.plot(x, slope*x+inercept, color='r')  # y=mx+b-
	plt.show()

	[inercept, slope] = gradient_descent(x, y, inercept, slope, learning_rate, iter_num)
	
	plt.scatter(x, y, marker = 'x', color='g')
	plt.plot(x, slope*x+inercept, color='r') # y=mx+b
	plt.show()
	
	print ("Final cost: ",cost_fun(x, y, inercept, slope))

if __name__ == '__main__':
	main()
