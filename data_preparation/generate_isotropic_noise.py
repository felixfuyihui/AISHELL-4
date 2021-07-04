import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import soundfile as sf
import linecache
import argparse
import os

speed_of_sound = 340  #m/s
def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()
# Hoth Noise Specifications
# For details, see p. 80 of
# http://studylib.net/doc/18787871/ieee-std-269-2001-draft-standard-methods-for-measuring
hoth_freqs = [
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
    2500, 3150, 4000, 5000, 6300, 8000
]
hoth_mag_db = [
    32.4, 30.9, 29.1, 27.6, 26, 24.4, 22.7, 21.1, 19.5, 17.8, 16.2, 14.6, 12.9,
    11.3, 9.6, 7.8, 5.4, 2.6, -1.3, -6.6
]
hoth_index_1000_hz = 10
hoth_index_4000_hz = 16
hoth_tolerance = 3  # +/- 3dB
# note that these values are values above, plus those extrapolated below 100Hz from Matlab's interp1 which can do extrapolation for cubic splines.
# these values and the Matlab interpolation command were gotten from Ivan Tashev's GenerateNoise.m matlab script
# these values are linear (not dB) and already noramalized wrt to value at 1000 Hz
#hoth_mag_extrap = [9.6982, 9.5348, 9.1337, 8.5567, 7.8652, 7.1208, 6.456542, 5.432503, 4.415704, 3.715352, 3.090295,
#                   2.570396, 2.113489, 1.757924, 1.462177, 1.202264, 1.000000, 0.831764, 0.683912, 0.568853, 0.467735,
#                   0.380189, 0.288403, 0.208930, 0.133352, 0.072444]
#hoth_freqs_extrap = [7.8125, 23.4375, 39.0625, 54.6875, 70.3125, 85.9375, 100, 125, 160, 200, 250, 315, 400, 500,
#                     630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]


def sample_sphere(num_points):
    # theta = elevation
    # phi = azimuth
    radius = 1
    theta = np.zeros([num_points])
    phi = np.zeros([num_points])
    for k in range(0, num_points, 1):
        h = -1 + 2 * k / (num_points - 1)
        phi[k] = np.arccos(h)
        if k == 0 or k == num_points - 1:
            theta[k] = 0
        else:
            theta[k] = np.mod(
                theta[k - 1] + 3.6 / np.sqrt(num_points * (1 - h * h)),
                2 * np.pi)

    loc_xyz = np.zeros([3, len(theta) * len(phi)])
    for k in range(0, num_points, 1):
        p = phi[k]
        t = theta[k]
        x = radius * np.sin(p) * np.cos(t)
        y = radius * np.sin(p) * np.sin(t)
        z = radius * np.cos(p)
        loc_xyz[:, k] = [x, y, z]
    return loc_xyz


def get_hoth_mag(samp_rate, fft_size):

    fft_size_by_2 = int(fft_size / 2)
    hoth_mag = np.asarray(hoth_mag_db) - hoth_mag_db[hoth_index_1000_hz]
    hoth_mag = np.power(10, hoth_mag / 20)
    hoth_w = 2 * np.pi * np.asarray(hoth_freqs) / samp_rate

    if (samp_rate == 16000):
        f = interp.interp1d(hoth_w,
                            hoth_mag,
                            kind='cubic',
                            bounds_error=False,
                            fill_value=(hoth_mag[0], hoth_mag[-1]))
    elif (samp_rate == 8000):
        f = interp.interp1d(hoth_w[0:hoth_index_4000_hz + 1],
                            hoth_mag[0:hoth_index_4000_hz + 1],
                            kind='cubic',
                            bounds_error=False,
                            fill_value=(hoth_mag[0],
                                        hoth_mag[hoth_index_4000_hz + 1]))
    else:
        RuntimeError('Can only generate Hoth noise for 16000 sampling rates!')

    w = 2 * np.pi * np.arange(0, fft_size_by_2 + 1, 1) / fft_size

    hoth_mag_interp = f(w)
    hoth_mag_interp[0] = 0  # skip DC (0 Hz)
    # add DC
    #hoth_mag_interp = np.insert(hoth_mag_interp, 0, 0)
    return hoth_mag_interp


def sample_circle(num_points):
    # phi = azimuth
    radius = 1
    phi = 2 * np.pi * np.arange(0, 1, 1 / num_points)
    loc_xyz = np.zeros([3, len(phi)])
    for k in range(0, num_points, 1):
        p = phi[k]
        x = radius * np.cos(p)
        y = radius * np.sin(p)
        z = 0
        loc_xyz[:, k] = [x, y, z]

    return loc_xyz


# Follows
# E.A.P. Habets and S. Gannot, Generating sensor signals in isotropic noise fields,
# Journal of the Acoustical Society of America, Vol. 122, Issue 6, pp. 3464-3470, Dec. 2007.
# added the spectral shaping to "Hoth" noise profile
def generate_isotropic_noise(mic_xyz,
                             N,
                             samp_rate=16000,
                             type='sph',
                             spectrum='hoth'):
    num_points = 64
    num_mics = mic_xyz.shape[0]
    fft_size = int(2**np.ceil(np.log2(N)))
    fft_size_by_2 = int(fft_size / 2)

    # calculate relative microphone positions wrt mic 1
    P_rel = np.zeros([num_mics, 3])
    for m in range(0, num_mics, 1):
        P_rel[m, :] = mic_xyz[m, :] - mic_xyz[0, :]

    # get locations uniformly sampled on a sphere
    if (type == 'sph'):
        loc_xyz = sample_sphere(num_points)
    elif (type == 'cyl'):
        loc_xyz = sample_circle(num_points)
    else:
        RuntimeError('type must be \'sph\' or \'cyl\'')

    if (spectrum == 'white'):
        g = 1
    elif (spectrum == 'hoth'):
        g = get_hoth_mag(samp_rate, fft_size)
    else:
        RuntimeError('spectrum must be \'white\' or \'hoth\'')

    # for each point, generate random noise in frequency domain and multiply by the steering vector
    w = 2 * np.pi * np.arange(0, fft_size_by_2 + 1, 1) / fft_size
    X = np.zeros([num_mics, fft_size_by_2 + 1], dtype=complex)

    for i in range(0, num_points, 1):
        X_this = g * (np.random.normal(0, 1, fft_size_by_2 + 1) +
                      1j * np.random.normal(0, 1, fft_size_by_2 + 1))
        X[0, :] = X[0, :] + X_this
        for m in range(1, num_mics, 1):
            delta = np.sum(P_rel[m, :] * loc_xyz[:, i])
            tau = delta * samp_rate / speed_of_sound
            X[m, :] = X[m, :] + X_this * np.exp(-1j * tau * w)

    X = X / np.sqrt(num_points)

    # transform to time domain
    X[:, 0] = np.sqrt(fft_size) * np.real(X[:, 0])
    X[:, fft_size_by_2] = np.sqrt(fft_size) * np.real(X[:, fft_size_by_2])
    X[:, 1:fft_size_by_2] = np.sqrt(fft_size_by_2) * X[:, 1:fft_size_by_2]

    n = np.fft.irfft(X, fft_size, axis=1)
    n = n[:, 0:N]
    n =n /np.max(np.abs(n))
    return n

def generate_isotropic_noise_fromsinglechannel(mic_xyz,
					                           noise,
					                           samp_rate=16000,
					                           type='sph',
					                           spectrum='hoth'):
    noise_wav = sf.read(noise)[0]
    N = noise_wav.shape[0]
    num_points = 64
    num_mics = mic_xyz.shape[0]
    fft_size = int(2**np.ceil(np.log2(N)))
    fft_size_by_2 = int(fft_size / 2)
    fft_noise_wav = np.fft.rfft(noise_wav, fft_size,axis=0)
    # fft_noise_wav = fft_noise_wav/np.max(np.abs(fft_noise_wav))
    # calculate relative microphone positions wrt mic 1
    P_rel = np.zeros([num_mics, 3])
    for m in range(0, num_mics, 1):
        P_rel[m, :] = mic_xyz[m, :] - mic_xyz[0, :]

    # get locations uniformly sampled on a sphere
    if (type == 'sph'):
        loc_xyz = sample_sphere(num_points)
    elif (type == 'cyl'):
        loc_xyz = sample_circle(num_points)
    else:
        RuntimeError('type must be \'sph\' or \'cyl\'')

    if (spectrum == 'white'):
        g = 1
    elif (spectrum == 'hoth'):
        g = get_hoth_mag(samp_rate, fft_size)
    else:
        RuntimeError('spectrum must be \'white\' or \'hoth\'')

    # for each point, generate random noise in frequency domain and multiply by the steering vector
    w = 2 * np.pi * np.arange(0, fft_size_by_2 + 1, 1) / fft_size
    X = np.zeros([num_mics, fft_size_by_2 + 1], dtype=complex)
    for m in range(0, num_mics, 1):
    	X[m, :] = X[m, :] + fft_noise_wav
    for i in range(0, num_points, 1):
        X_this = g * (np.random.normal(0, 1, fft_size_by_2 + 1) +
                      1j * np.random.normal(0, 1, fft_size_by_2 + 1))
        X[0, :] = X[0, :] + X_this
        for m in range(1, num_mics, 1):
            delta = np.sum(P_rel[m, :] * loc_xyz[:, i])
            tau = delta * samp_rate / speed_of_sound
            X[m, :] = X[m, :] + X_this * np.exp(-1j * tau * w)

    X = X / np.sqrt(num_points)

    # transform to time domain
    X[:, 0] = np.sqrt(fft_size) * np.real(X[:, 0])
    X[:, fft_size_by_2] = np.sqrt(fft_size) * np.real(X[:, fft_size_by_2])
    X[:, 1:fft_size_by_2] = np.sqrt(fft_size_by_2) * X[:, 1:fft_size_by_2]

    n = np.fft.irfft(X, fft_size, axis=1)
    n = n[:, 0:N]
    n = n * np.max(np.abs(noise_wav)) / np.max(np.abs(n))
    return n


def run(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    mic_distance = 0.05
    baseangle = np.pi/8

    mic = np.array([[mic_distance*np.cos(8*baseangle), mic_distance*np.sin(baseangle*8), 0],
                   [mic_distance*np.cos(7*baseangle), mic_distance*np.sin(baseangle*7), 0],
                   [mic_distance*np.cos(6*baseangle), mic_distance*np.sin(baseangle*6), 0],
                   [mic_distance*np.cos(5*baseangle), mic_distance*np.sin(baseangle*5), 0],
                   [mic_distance*np.cos(4*baseangle), mic_distance*np.sin(baseangle*4), 0],
                   [mic_distance*np.cos(3*baseangle), mic_distance*np.sin(baseangle*3), 0],
                   [mic_distance*np.cos(2*baseangle), mic_distance*np.sin(baseangle*2), 0],
                   [mic_distance*np.cos(1*baseangle), mic_distance*np.sin(baseangle*1), 0],],dtype=np.float32)
    for i in range(args.wavnum):
        y = generate_isotropic_noise(mic, 960000)
        y = y.transpose()
        sf.write(args.output_dir+'/isotropic_'+str(i+1)+'.wav',y,16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        type=str,
                        help="toutput_dir for data",
                        default="isotropic_noise")
    parser.add_argument("--wavnum",
                        type=int,
                        help="total number of simulated wavs",
                        default=200)
    args = parser.parse_args()
    run(args)