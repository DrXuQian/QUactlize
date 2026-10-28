"""Product correctness-test support; no performance campaign."""
import ctypes as C
from quactlize.runtime.native import checked
class Replay:
    def __init__(self, sdk, stream, launch, repeats):
        self.sdk, self.stream = sdk, stream
        self.graph, self.instance = C.c_void_p(), C.c_void_p()
        checked(sdk.lib.hggcStreamBeginCapture(stream, 0), "begin capture")
        try:
            for _ in range(repeats):
                checked(launch(), "capture selected call")
        except BaseException:
            sdk.lib.hggcStreamEndCapture(stream, C.byref(self.graph))
            self.close()
            raise
        checked(
            sdk.lib.hggcStreamEndCapture(stream, C.byref(self.graph)), "end capture"
        )
        try:
            checked(
                sdk.lib.hggcGraphInstantiateWithFlags(
                    C.byref(self.instance), self.graph, 0
                ),
                "instantiate",
            )
        except BaseException:
            self.close()
            raise

    def __call__(self):
        return self.sdk.lib.hggcGraphLaunch(self.instance, self.stream)

    def close(self):
        if self.instance:
            checked(
                self.sdk.lib.hggcGraphExecDestroy(self.instance),
                "destroy graph instance",
            )
            self.instance = C.c_void_p()
        if self.graph:
            checked(self.sdk.lib.hggcGraphDestroy(self.graph), "destroy graph")
            self.graph = C.c_void_p()
