import sys, math, os, logging
import cv2
import numpy as np
import pyrtools as pt
import platform
import glob
import ctypes
import scipy
import matplotlib.pyplot as plt
from skimage import color


### RGB2YIQ2 AND YIQ2RGB


def rgb2ntsc(frame):
    YIQ = np.ndarray(frame.shape)

    YIQ[:, :, 0] = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    YIQ[:, :, 1] = 0.59590059 * frame[:, :, 0] + (-0.27455667) * frame[:, :, 1] + (-0.32134392) * frame[:, :, 2]
    YIQ[:, :, 2] = 0.21153661 * frame[:, :, 0] + (-0.52273617) * frame[:, :, 1] + 0.31119955 * frame[:, :, 2]
    return YIQ


def ntsc2rgb(frame):
    RGB = np.ndarray(frame.shape)
    RGB[:, :, 0] = 1.00000001 * frame[:, :, 0] + 0.95598634 * frame[:, :, 1] + 0.6208248 * frame[:, :, 2]
    RGB[:, :, 1] = 0.99999999 * frame[:, :, 0] + (-0.27201283) * frame[:, :, 1] + (-0.64720424) * frame[:, :, 2]
    RGB[:, :, 2] = 1.00000002 * frame[:, :, 0] + (-1.10674021) * frame[:, :, 1] + 1.70423049 * frame[:, :, 2]
    return RGB


### Ideal_bandpassing


def shiftdim(x, n):
    return x.transpose(np.roll(range(x.ndim), -n))


def repmat(a,m):
    #First, pad out a so it has same dimensionality as m
    for i in range(0,m.ndim-a.ndim):
        a = np.expand_dims(a,1)
    #Now just use numpy tile and return result
    return np.tile(a,m.shape)


def ideal_bandpassing(input, dim, wl, wh, samplingRate):
    # if dim is greater than the dimensionality (2d, 3d etc) of the input, quit
    if (dim > len(input.shape)):
        print('Exceed maximum dimension')
        return

    # This has the effect that input_shifted[0] = input[dim]
    input_shifted = shiftdim(input, dim - 1)

    # Put the dimensions of input_shifted in a 1d array
    Dimensions = np.asarray(input_shifted.shape)

    # how many things in the first dimension of input_shifted
    n = Dimensions[0]

    # get the dimensionality (eg. 2d, 3d etc) of input_shifted
    dn = input_shifted.ndim

    # creates a vector [1,...,n], the same length as the first dimension of input_shifted
    Freq = np.arange(1.0, n + 1)

    # Equivalent in python: Freq = (Freq-1)/n*samplingRate
    Freq = Freq / n * samplingRate

    # Create boolean mask same size as Freq, true in between the frequency limits wl,wh
    mask = (Freq > wl) & (Freq < wh)

    Dimensions[0] = 1
    mask = repmat(mask, np.ndarray(Dimensions))

    # F = fft(X,[],dim) and F = fft(X,n,dim) applies the FFT operation across the dimension dim.
    # Python: F = np.fftn(a=input_shifted,axes=0)
    F = np.fft.fftn(a=input_shifted, axes=[0])

    # So we are indexing array F using boolean not mask, and setting those values of F to zero, so the others pass thru
    # Python: F[ np.logical_not(mask) ]
    F[np.logical_not(mask)] = 0

    # Get the real part of the inverse fourier transform of the filtered input
    filtered = np.fft.ifftn(a=F, axes=[0]).real

    filtered = filtered.astype(np.float32)

    filtered = shiftdim(filtered, dn - (dim - 1))

    return filtered
#
# def ideal_bandpassing(input, dim, wl, wh, fs):
#     if dim > np.asarray(input.shape).ndim:
#         print('Exceed maximum dimension')
#     input_shifted = shiftdim(input, dim - 1)  # need to implement shift_dim
#     dimension = np.asarray(input_shifted.shape)
#
#     n = dimension[0]
#     dn = input_shifted.ndim
#
#     Freq = np.arange(n)
#     Freq = Freq / n * fs  # removed minus 1 because the array start at 0 and matlab starts at 1 so in matalb yousould subtract by 1
#     mask = (Freq > wl) & (Freq < wh)
#     dimension[0] = 1
#     mask = mask.flatten('F')
#     # tmp = repmat(mask,dimension)
#     mask = np.tile(mask, np.ndarray(dimension))
#     F = np.fft.fft(input_shifted, axis=0)
#     mask.resize(F.shape)
#     F[~mask] = 0  # dont know what is this
#
#     filtered = np.fft.ifft(a=F, axis=0).real
#     filtered = filtered.astype(np.float32)
#
#     filtered = shiftdim(filtered, dn - (dim - 1))
#
#     return filtered


### build_gdown_stack


def build_gdown_stack(vid_file, start_index, end_index, level):
    vid = cv2.VideoCapture(vid_file)
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_channels = 3

    suc, temp = vid.read()
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    # frame = cv2.normalize(temp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # im2double
    frame = im2double(temp)
    frame = color.rgb2yiq(frame)
    frame = rgb2ntsc(frame)

    blurred = blur_dn_clr(frame, level)  # need to implement this

    gdown_stack = np.zeros((end_index - start_index + 1, blurred.shape[0], blurred.shape[1], blurred.shape[2]))
    gdown_stack[0, :, :, :] = blurred

    for k in range(start_index, end_index + 1):
        succ, temp = vid.read()
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        frame = im2double(temp)
        # frame = cv2.normalize(temp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # im2double
        # frame = rgb2ntsc(frame)
        frame = color.rgb2yiq(frame)
        blurred = blur_dn_clr(frame, level)
        gdown_stack[k, :, :, :] = blurred

    return gdown_stack


### amplify_spatial_Gdown_temporal_ideal

def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype


def amplify_spatial_Gdown_temporal_ideal(vid_file, out_file, alpha, level, fl, fh, fs, chrom_attenuation):
    out_name = "out2.avi"
    vid = cv2.VideoCapture(vid_file)
    fr = vid.get(cv2.CAP_PROP_FPS)
    len_ = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_index = 0
    end_index = len_ - 10

    cap_size = (vid_width, vid_height)  # this is the size of my source video
    vid_out = cv2.VideoWriter()
    fourcc = vid_out.fourcc('j', 'p', 'e', 'g')  # note the lower case
    success = vid_out.open(out_name, fourcc, fr, cap_size, True)

    logging.info('Spatial filtering...')
    gdown_stack = build_gdown_stack(vid_file, start_index, end_index, level)
    logging.info('Finished')

    logging.info('Temporal filtering...')
    filtered_stack = ideal_bandpassing(gdown_stack, 1, fl, fh, fs)
    logging.info('Finished')

    # amplify
    filtered_stack[:, :, :, 0] = filtered_stack[:, :, :, 0] * alpha
    filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha * chrom_attenuation
    filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chrom_attenuation

    logging.info('Rendering...')

    for k in range(start_index, end_index + 1):
        succ, temp = vid.read()
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        # frame = cv2.normalize(temp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # im2double
        frame = im2double(temp)
        frame = color.rgb2yiq(frame)
        filtered = (filtered_stack[k, :, :, :]).squeeze()
        filtered = cv2.resize(filtered, (vid_width, vid_height), 0, 0, cv2.INTER_LINEAR)
        filtered = filtered + frame
        frame = color.yiq2rgb(filtered)
        frame *= 255
        frame = np.clip(frame, 0, 255)
        frame = cv2.convertScaleAbs(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid_out.write(frame)

    logging.info('Finished')
    vid_out.release()
    vid.release()


def binomial_filter(sz):
    if sz < 2:
        logging.warning('size argument must be larger than 1')
    kernel = [0.5, 0.5]
    for n in range(0, sz - 2):
        kernel = np.convolve([0.5, 0.5], kernel)
    return kernel


def named_filter(name):
    if name[:5] == 'binom':
        kernel = np.sqrt(2) * binomial_filter(int(name[5:]))
    elif name == 'qmf5':
        kernel = np.asarray((-0.076103, 0.3535534, 0.8593118, 0.3535534, -0.076103))
    elif name == 'qmf9':
        kernel = np.asarray((0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934,
                             0.41472545, -0.073386624, -0.060944743, 0.02807382))
    elif name == 'qmf13':
        kernel = np.asarray((-0.014556438, 0.021651438, 0.039045125, -0.09800052,
                             -0.057827797, 0.42995453, 0.7737113, 0.42995453, -0.057827797,
                             -0.09800052, 0.039045125, 0.021651438, -0.014556438))
    elif name == 'qmf8':
        kernel = np.sqrt(2) * np.asarray((0.00938715, -0.07065183, 0.06942827, 0.4899808,
                                          0.4899808, 0.06942827, -0.07065183, 0.00938715))
    elif name == 'qmf12':
        kernel = np.sqrt(2) * np.asarray((-0.003809699, 0.01885659, -0.002710326, -0.08469594,
                                          0.08846992, 0.4843894, 0.4843894, 0.08846992, -0.08469594, -0.002710326,
                                          0.01885659, -0.003809699))
    elif name == 'qmf16':
        kernel = np.sqrt(2) * np.asarray((0.001050167, -0.005054526, -0.002589756, 0.0276414, -0.009666376,
                                          -0.09039223, 0.09779817, 0.4810284, 0.4810284, 0.09779817, -0.09039223,
                                          -0.009666376,
                                          0.0276414, -0.002589756, -0.005054526, 0.001050167))
    elif name == 'haar':
        kernel = np.asarray((1, 1)) / np.sqrt(2)
    elif name == 'daub2':
        kernel = np.asarray((0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551))
    elif name == 'daub3':
        kernel = np.asarray((0.332670552950, 0.806891509311, 0.459877502118, -0.135011020010,
                             -0.085441273882, 0.035226291882))
    elif name == 'daub4':
        kernel = np.asarray((0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
                             -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785))
    elif name == 'gauss5':  # for backward-compatibility
        kernel = np.sqrt(2) * np.asarray((0.0625, 0.25, 0.375, 0.25, 0.0625))
    elif name == 'gauss3':  # for backward-compatibility
        kernel = np.sqrt(2) * np.asarray((0.25, 0.5, 0.25))
    else:
        logging.warning('Bad filter name: ', name)
        return

    return kernel


def parse_filter(filt, normalize=True):
    if isinstance(filt, str):
        filt = named_filter(filt)

    elif isinstance(filt, np.ndarray) or isinstance(filt, list) or isinstance(filt, tuple):
        filt = np.array(filt)
        if filt.ndim == 1:
            filt = np.reshape(filt, (filt.shape[0], 1))
        elif filt.ndim == 2 and filt.shape[0] == 1:
            filt = np.reshape(filt, (-1, 1))

    # TODO expand normalization options
    if normalize:
        filt = filt / filt.sum()

    return filt


# def blur_dn(image, n_levels=1, filt='binom5'):
#     if image.ndim == 1:
#         image = image.reshape(-1, 1)
#
#     filt = parse_filter(filt)
#     filt = filt / np.sum(filt.flatten('F'))  # normalize
#     if n_levels > 1:
#         image = blur_dn(image, n_levels - 1, filt)
#
#     if n_levels >= 1:
#         if image.shape[1] == 1:
#             # 1D image [M, 1] and 1D filter [N, 1]
#             res = corr_dn(image=image, filt=filt, step=(2, 1))
#
#         elif image.shape[0] == 1:
#             # 1D image [1, M] and 1D filter [N, 1]
#             res = corr_dn(image=image, filt=filt.T, step=(1, 2))
#
#         elif filt.shape[1] == 1:
#             # 2D image and 1D filter [N, 1]
#             res = corr_dn(image=image, filt=filt, step=(2, 1))
#             res = corr_dn(image=res, filt=filt.T, step=(1, 2))
#
#         else:
#             # 2D image and 2D filter
#             res = corr_dn(image=image, filt=filt, step=(2, 2))
#
#     else:
#         res = image
#
#     return res


def blur_dn(im, nlevs=1, filt='binom5'):
    if type(filt) == str:
        filt = named_filter(filt)
    filt = filt / np.sum(filt.flatten('F'))
    if nlevs > 1:
        im = blur_dn(im, nlevs - 1, filt)

    if nlevs >= 1:
        if np.asarray(im.shape).ndim == 1:
            if np.asarray(filt.shape).ndim != 1:
                logging.warning('Cant apply 2D filter to 1D signal')
                return
            if im.ndim == 1:
                filt = filt.flatten('F')
            else:
                filt = filt.flatten('F')
                filt = np.conjugate(filt).T
                filt = filt.reshape(-1, 1)
            res = corr_dn(im, filt, 'reflect1', tuple(map(lambda x: int(not x == 1) + 1, im.shape)))
        elif np.asarray(filt.shape).ndim == 1:
            filt = filt.flatten("F")
            res = corr_dn(im, filt, 'reflect1', (2, 1))
            res = corr_dn(res, filt, 'reflect1', (1, 2))
        else:
            res = corr_dn(im, filt, 'reflect1', (2, 2))
    else:
        res = im
    return res


def blur_dn_clr(im, n_levs=1, filt='binom5'):
    # tmp = pt.blurDn(im[:,:,0],nlevs,filt)
    # tmp = pt.blurDn(im[:, :, 0].copy(), n_levs, filt)
    tmp = blur_dn(im[:, :, 0].copy(), n_levs, filt)
    out = np.zeros((tmp.shape[0], tmp.shape[1], im.shape[2]))
    out[:, :, 0] = tmp
    for clr in range(1, im.shape[2]):
        # out[:, :, clr] = pt.blur_dn(im[:, :, clr], n_levs, filt)
        out[:, :, clr] = blur_dn(im[:, :, clr], n_levs, filt)
    return out


# def corr_dn(nhls, phls, nrhs, prhs):
#     logging.info("Parameters received:\n" + str(nhls) + "\n" + str(phls) +
#                  "\n" + str(nrhs) + "\n" + str(prhs))
#     x_start = 1
#     x_step = 1
#     y_start = 1
#     y_step = 1
#
#     edges = 'reflect1'
#     edges = 'reflect1'
#
#     logging.info(prhs[0])
#     arg0 = prhs[0]
#     image = arg0
#     x_idim = int(arg0.shape[0])
#     y_idim = int(arg0.shape[1])
#
#     arg1 = prhs[1]
#     flit = arg1.data
#     x_fdim = int(arg1.shape[0])
#     y_fdim = int(arg1.shape[1])
#
#     x_start -= 1
#     y_start -= 1
#
#     x_rdim = (x_stop-x_start+x_step-1) / x_step
#     y_rdim = (y_stop-y_start+y_step-1) / y_step
#
#     plhs[0] = np.zeros((x_rdim, y_rdim))
#     result = plhs[0]
#
#     if edges == "circular":
#         pt.internal_wrap_reduce(image, x_idim, y_idim, flit, x_fdim, y_fdim,
#                x_start, x_step, y_start, y_step,
#                result)
#     else:
#         pt.internal_reduce(image, x_idim, y_idim, flit, x_fdim, y_fdim,
#                x_start, x_step, y_start, y_step,
#                result)
#     return result

# def corr_dn(im,file,edges,step = [1,1],start = [1,1]):
#     stop = len(im)
#     filt = filt[filt.shape[0]:-1:0,filt.shape[1]:-1:0]
#     temp = rconv2(im,filt)
#
#     res = tmp[start[0]:step[0]:stop[0],start[1]:step[1]:stop[1]]
#     return res

def corr_dn(image, filt, edge_type='reflect1', step=(1, 1), start=(0, 0),
            stop=None):
    image = image.copy().astype(float)
    filt = filt.copy().astype(float)

    if image.shape[0] < filt.shape[0] or image.shape[1] < filt.shape[1]:
        raise Exception("Signal smaller than filter in corresponding dimension: ", image.shape, filt.shape,
                        " see parse filter")

    if edge_type not in ['circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute']:
        raise Exception("Don't know how to do convolution with edge_type %s!" % edge_type)

    if filt.ndim == 1:
        filt = filt.reshape(1, -1)

    if stop is None:
        stop = (image.shape[0], image.shape[1])

    rxsz = len(range(start[0], stop[0], step[0]))
    rysz = len(range(start[1], stop[1], step[1]))
    result = np.zeros((rxsz, rysz))

    if edge_type == 'circular':
        lib.internal_wrap_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 image.shape[1], image.shape[0],
                                 filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 filt.shape[1], filt.shape[0],
                                 start[1], step[1], stop[1], start[0], step[0],
                                 stop[0],
                                 result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        tmp = np.zeros((filt.shape[0], filt.shape[1]))
        lib.internal_reduce(image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            image.shape[1], image.shape[0],
                            filt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            filt.shape[1], filt.shape[0],
                            start[1], step[1], stop[1], start[0], step[0],
                            stop[0],
                            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            edge_type.encode('ascii'))

    return result


def reconv2(a, b, ctr):
    if len(a[1]) >= len(b[1]) and len(a[2]) >= len(b[2]):
        large = a
        small = b
    elif len(a[1]) <= len(b[1]) and len(a[2]) <= len(b[2]):
        large = b
        small = a
    else:
        raise Exception("one arg must be larger than the other in both dimensions!")

    ly = len(large[1])
    lx = len(large[2])
    sy = len(small[1])
    sx = len(small[2])

    """
    These values are one less than the index of the small mtx that falls on
    the border pixel of the large matrix when computing the first
    convolution response sample:
    """
    sy2 = math.floor((sy + ctr - 1) / 2)
    sx2 = math.floor((sx + ctr - 1) / 2)

    # pad with reflected copies
    clarge = np.asarray(np.block(large[sy - sy2:-1:2, sx - sx2:-1:2], large[sy - sy2:-1:2, :],
                                 large[sy - sy2:-1:2, lx - 1:-1:lx - sx2]),
                        np.block(large[:, sx - sx2:-1:2], large, large[:, lx - 1:-1:lx - sx2]),
                        np.block(large[ly - 1:-1:ly - sy2, sx - sx2:-1:2], large[ly - 1:-1:ly - sy2, :],
                                 large[ly - 1:-1:ly - sy2, lx - 1:-1:lx - sx2]))

    c = scipy.signal.convolve2d(clarge, small, boundary='valid', mode='same')

    return c


# Running the algorithm:


FORMAT = '[%(asctime)s] [%(levelname)s] [%(funcName)s] [%(lineno)d] : %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logging.info("Starting ...")
if platform.system() == "Windows":
    seperator = "\\"
else:
    seperator = "/"

dir = "perry-all-2"
# should be a parameter of the engine
dataset_location = ".." + seperator + "dataset" + seperator + "good_sync" + seperator
specific_dir = dir
video_location = dataset_location + specific_dir + seperator + "test1.mp4"

libpath = glob.glob(
    "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/pyrtools/pyramids/c/wrapConv.cpython-38-darwin.so")
logging.info("libpath:" + str(libpath))

# load the c library
if len(libpath) > 0:
    lib = ctypes.cdll.LoadLibrary(libpath[0])
else:
    logging.warning("Can't load in C code, something went wrong in your install!")

# hpyer params here were taken from the matlab implementation but might need to be changed (there were several options and I took the face option)
amplify_spatial_Gdown_temporal_ideal(video_location, "/", 50, 4, 50 / 60, 60 / 60, 30, 1)
