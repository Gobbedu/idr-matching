import numpy
import math
import bob.ip.gabor
import bob.sp

# create exemplary image
image = numpy.zeros((128,128), numpy.uint8)
image[32:96,32:96] = 255

# create Gabor wavelet
wavelet = bob.ip.gabor.Wavelet(resolution = (128, 128), frequency = (math.pi/8., 0))

# compute wavelet transform in frequency domain
freq_image = bob.sp.fft(image.astype(numpy.complex128))
transformed_freq_image = wavelet.transform(freq_image)
transformed_image = bob.sp.ifft(transformed_freq_image)

# get layers of the image
real_image = numpy.real(transformed_image)
abs_image = numpy.abs(transformed_image)

# get the wavelet in spatial domain
spat_wavelet = bob.sp.ifft(wavelet.wavelet.astype(numpy.complex128))
real_wavelet = numpy.real(spat_wavelet)
# align wavelet to show it centered
aligned_wavelet = numpy.roll(numpy.roll(real_wavelet, 64, 0), 64, 1)

# create figure
from matplotlib import pyplot
pyplot.figure(figsize=(20,5))
pyplot.subplot(141)
pyplot.imshow(image, cmap='gray')
pyplot.title("Original image")

pyplot.subplot(142)
pyplot.imshow(aligned_wavelet, cmap='gray')
pyplot.title("Gabor wavelet")

pyplot.subplot(143)
pyplot.imshow(real_image, cmap='gray')
pyplot.title("Real part")

pyplot.subplot(144)
pyplot.imshow(abs_image, cmap='gray')
pyplot.title("Abs part")

pyplot.show()

