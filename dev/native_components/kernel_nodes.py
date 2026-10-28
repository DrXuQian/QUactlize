"""Opaque node holders: never reinterpret a Driver function as a Runtime stub."""
import ctypes as C


class KernelInfo(C.Structure):
    _fields_ = [('grid',C.c_uint32*3),('block',C.c_uint32*3),('shared',C.c_uint32),
                ('driver',C.c_uint32),('symbol',C.c_char_p)]


class KernelNodes:
    def __init__(self, library):
        if C.sizeof(KernelInfo)!=40 or KernelInfo.symbol.offset!=32:
            raise ValueError('kernel-node bridge host ABI differs')
        self.library=library
        def bind(name,args,result=C.c_int):
            fn=getattr(library,'nc_kernel_'+name);fn.argtypes,fn.restype=args,result;return fn
        self.capture_fn=bind('capture',[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(KernelInfo)])
        self.add_fn=bind('add',[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p,C.c_void_p])
        self.destroy=bind('destroy',[C.c_void_p],None)
        self.error=library.nc_error;self.error.argtypes=[];self.error.restype=C.c_char_p

    def check(self,rc):
        if rc:raise RuntimeError('kernel-node bridge: '+self.error().decode(errors='replace'))

    def capture(self,node):
        handle=C.c_void_p();info=KernelInfo()
        self.check(self.capture_fn(node,C.byref(handle),C.byref(info)))
        try:
            if not handle.value or not info.symbol or any(x==0 for x in (*info.grid,*info.block)):
                raise ValueError('incomplete kernel-node description')
            return handle,dict(symbol=info.symbol.decode(),grid=list(info.grid),block=list(info.block),
                shared_bytes=info.shared,parameter_api='DRIVER_CUFUNCTION' if info.driver else 'RUNTIME_HOST_STUB')
        except BaseException:
            self.destroy(handle);raise

    def add(self,handle,result,graph,previous):
        self.check(self.add_fn(handle,result,graph,previous))
