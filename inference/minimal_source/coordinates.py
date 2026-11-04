import numpy as np
_MASK=(1<<64)-1

def _key(value):
    if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= _MASK:
        raise ValueError('Keys must be unsigned 64-bit integers')
    return np.uint64(value)

def _mix(x):
    with np.errstate(over='ignore'):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        return x ^ (x >> np.uint64(31))

def open_uniform_from_bits(bits):


    bits = np.asarray(bits, dtype=np.uint64)
    return ((bits >> np.uint64(12)).astype(np.float64) + .5) * 2.**-52

def coordinate_uniform(coordinates, seed=7103, sample_id=0, image_id=0, stream=0):
    coordinates = np.asarray(coordinates)
    if coordinates.dtype.kind not in 'iu' or np.any(coordinates < 0):
        raise ValueError('Absolute coordinate IDs must be nonnegative integers')
    key = _mix(_key(seed)) ^ _mix(_key(sample_id) ^ np.uint64(0x243f6a8885a308d3))
    key ^= _mix(_key(image_id) ^ np.uint64(0x13198a2e03707344))
    key ^= _mix(_key(stream) ^ np.uint64(0xa4093822299f31d0))
    return open_uniform_from_bits(_mix(coordinates.astype(np.uint64) ^ key))
