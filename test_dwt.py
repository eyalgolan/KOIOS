import pywt
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet as MCD
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_wavelet
from matplotlib import pyplot as plt


def multivariate_Video_Signal(signal):
    """
    Algorithm 1
    :param signal: is a 2D signal. One dimension the signal from the forehead, and second is from above the upper lip
    :return: multivariate PPG!
    """
    H, J = 1, 4  # currently H equals 1 but in the future for a real signal it should be 2.
    dwt_res = [(None, signal)]
    for i in range(0, J):
        approx_i, det_i = pywt.dwt2(dwt_res[i][1], 'sym2')
        dwt_res.append((det_i, approx_i))

    first_details = (dwt_res[1][0]).reshape(-1, 1)
    noise_matrix = MCD(random_state=0).fit(first_details).covariance_
    v, diagonal_mat, v_transpose = np.linalg.svd(
        noise_matrix)
    xi_hats = []
    xi_js = []
    for j in range(1, J + 1):
        xi_j = np.dot(dwt_res[1][0], diagonal_mat)
        xi_js.append(xi_j)
        for h in range(1, H + 1):
            n = len(signal[0])
            val_h = np.sqrt(2 * diagonal_mat.diagonal() * np.log(n))
            xi_hat = pywt.threshold(data=xi_j[h], value=val_h)
            xi_hats.append(xi_hat)
    phi = None
    pca = PCA()
    phi_norm = StandardScaler().fit_transform(phi)
    pca.fit(phi_norm)
    phi_hat = pca.transform(phi_norm)
    xi_hat = None
    multivariate_ppg = pywt.idwt2(phi_hat, xi_hat, 'sym2')
    return multivariate_ppg


x = pywt.data.ecg().astype(float) / 256

sigma = 0.05
x_noist = x + sigma * np.random.randn(x.size)

multivariate_Video_Signal(x_noist)

#
# x_denoise = denoise_wavelet(x_noist, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8',
#                             rescale_sigma='True')

# plt.figure(figsize=(20, 10), dpi=100)
# plt.plot(x_noist)
# plt.plot(x_denoise)
