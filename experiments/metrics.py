import numpy as np

def PSNR(ground_truth, denoised, batch_size):
    image_size = ground_truth.shape[1] * ground_truth.shape[2]
    loss_mse = np.sum((np.power(denoised - ground_truth, 2))) / (batch_size * image_size)
    loss_psnr = 20 * np.log10(1.0 / np.sqrt(loss_mse))
    return loss_psnr



