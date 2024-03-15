import pdb
import numpy as np
import matplotlib.pyplot as plt

# Generate a random walk
def random_walk(n):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i-1] + np.random.normal(loc=0.1, scale=1.0, size=None)
    return x


def main():

    # Generate N random walks
    N = 1000
    n = 1000
    X = np.zeros((N, n))
    for i in range(N):
        X[i] = random_walk(n)

    # Normalize the random walks
    X = X / np.std(X, axis=1).reshape(-1,1)
        
    
    # Add noise to the random walks
    X_noisy = X.copy()
    for i in range(100):
        X_noisy += np.random.normal(loc=0.01, scale=0.1, size=(N,n))

    # Smooth the random walks using fourier transform
    X_fft = np.fft.rfft(X_noisy, axis=1)
    X_fft[:, 10:] = 0
    X_fourier = np.fft.irfft(X_fft, axis=1)

    # Smooth time series using a moving average
    X_smooth = np.zeros((N,n))
    for i in range(N):
        X_smooth[i] = np.convolve(X_noisy[i], np.ones(15)/15, mode='same')
        



    # Plot the first 10 random walks
    plt.figure()
    plt.subplot(1,4,1)
    for i in range(1):
        plt.plot(X_noisy[i], label='Noisy', alpha=0.5)
        plt.plot(X[i], label='Original', linewidth=2)
        plt.plot(X_fourier[i], label='Fourier', linewidth=3)
        plt.plot(X_smooth[i], label='Moving average', linewidth=3)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Random Walks')


    # Plot the mean and variance over time of the random walks
    plt.subplot(1,4,2)
    plt.plot(np.mean(X, axis=0), label='Mean')
    plt.plot(np.var(X, axis=0), label='Variance')
    plt.plot(np.mean(X_fourier, axis=0), label='Smoothed Mean')
    plt.plot(np.var(X_fourier, axis=0), label='Smoothed Variance')
    plt.plot(np.mean(X_smooth, axis=0), label='Moving Average Mean')
    plt.plot(np.var(X_smooth, axis=0), label='Moving Average Variance')
    plt.title('Mean and Variance')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()

    # Approximate gaussian distribution
    gauss_mean = np.mean(X.flatten())
    gauss_var = np.var(X.flatten()) 

    x_plot = np.linspace(-5, 5, 1000)
    gaussian = np.exp(-0.5*(x_plot - gauss_mean)**2 / gauss_var)
    gaussian = gaussian / np.sqrt(2*gauss_var*np.pi)


    # Plot the histogram of all the time series at all steps
    plt.subplot(1,4,3)
    plt.hist(X[0:10].flatten(), bins=50, density=True, label='Original', alpha=0.5)
    plt.hist(X_fourier[0:10].flatten(), bins=50, density=True, alpha=0.5, label='Fourier')
    plt.hist(X_smooth[0:10].flatten(), bins=50, density=True, alpha=0.5, label='Moving Average')
    plt.plot(x_plot, gaussian, label='Gaussian')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.title('Histogram of final positions')
    
    # Plot the cumulative distribution function of all the time series at all steps
    plt.subplot(1,4,4)
    plt.hist(X[0:10].flatten(), bins=50, density=True, cumulative=True, label='Original', alpha=0.5)
    plt.hist(X_fourier[0:10].flatten(), bins=50, density=True, cumulative=True, alpha=0.5, label='Fourier')
    plt.hist(X_smooth[0:10].flatten(), bins=50, density=True, cumulative=True, alpha=0.5, label='Moving Average')
    #plt.plot(x_plot, 0.5*(1+np.erf((x_plot-gauss_mean)/np.sqrt(2*gauss_var))), label='Gaussian')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.title('Cumulative distribution of final positions')
    plt.legend()
    plt.show()


    # Compute first four moments of the true distribution
    mean = np.mean(X.flatten())
    var = np.var(X.flatten())
    skew = np.mean((X.flatten() - mean)**3) / var**(3/2)
    kurt = np.mean((X.flatten() - mean)**4) / var**2 - 3

    # Compute first four moments of the Fourier smoothed distribution
    mean_fourier = np.mean(X_fourier.flatten())
    var_fourier = np.var(X_fourier.flatten())
    skew_fourier = np.mean((X_fourier.flatten() - mean_fourier)**3) / var_fourier**(3/2)
    kurt_fourier = np.mean((X_fourier.flatten() - mean_fourier)**4) / var_fourier**2 - 3

    # Compute first four moments of the moving average smoothed distribution
    mean_smooth = np.mean(X_smooth.flatten())
    var_smooth = np.var(X_smooth.flatten())
    skew_smooth = np.mean((X_smooth.flatten() - mean_smooth)**3) / var_smooth**(3/2)
    kurt_smooth = np.mean((X_smooth.flatten() - mean_smooth)**4) / var_smooth**2 - 3

    # Plot the error of first four moments in bar plot
    # Make sure the bars are next to each other and NOT on top of each other using offset
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.bar([1,2,3,4], [mean, var, skew, kurt], label='Original', width=0.1)
    plt.bar([1.1,2.1,3.1,4.1], [mean_fourier, var_fourier, skew_fourier, kurt_fourier], label='Fourier', width=0.1)
    plt.bar([1.2,2.2,3.2,4.2], [mean_smooth, var_smooth, skew_smooth, kurt_smooth], label='Moving Average', width=0.1)
    plt.xticks([1,2,3,4], ['Mean', 'Variance', 'Skewness', 'Kurtosis'])
    plt.ylabel('Value')
    plt.xlabel('Moment')
    plt.legend()
    plt.title('First four moments')

    # Plot the absolute error of first four moments in bar plot
    # Make sure the bars are next to each other and NOT on top of each other using offset
    plt.subplot(1,2,2)
    plt.bar([1.1,2.1,3.1,4.1], [np.abs(mean-mean_fourier), np.abs(var-var_fourier), np.abs(skew-skew_fourier), np.abs(kurt-kurt_fourier)], label='Fourier', width=0.1)
    plt.bar([1.2,2.2,3.2,4.2], [np.abs(mean-mean_smooth), np.abs(var-var_smooth), np.abs(skew-skew_smooth), np.abs(kurt-kurt_smooth)], label='Moving Average', width=0.1)
    plt.xticks([1,2,3,4], ['Mean', 'Variance', 'Skewness', 'Kurtosis'])
    plt.ylabel('Value')
    plt.xlabel('Moment')
    plt.legend()
    plt.title('Error of first four moments')
    plt.show()
    

    







    return 0


if __name__ == '__main__':
    main()



