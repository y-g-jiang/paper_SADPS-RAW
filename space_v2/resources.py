import ctypes, gc, os, time

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
        ['total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','extended']]

def available_commit():
    if os.name!='nt': return float('inf')
    s=MemoryStatus(); s.length=ctypes.sizeof(s)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.available_commit

def wait_headroom(gib=2):
    started=time.monotonic(); reported=False
    while available_commit()<gib*1024**3:
        if not reported:
            print(f'WAIT_RESOURCE: system commit headroom below {gib} GiB; preserving other processes',flush=True); reported=True
        gc.collect(); time.sleep(2)
    return time.monotonic()-started
