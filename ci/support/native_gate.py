"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import ctypes as C
import hashlib
import json
import numpy as np
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import arrangement, bind
from quactlize.runtime.native import Call, checked
from quactlize.runtime.tuning import Request
from ci.support.gemv import Resources, compare, metadata_oracle
from ci.support.graph_api import profiles
class OutputFailure(ValueError):
    def __init__(self, proof):
        self.proof = proof
        super().__init__("selected output check failed: " + json.dumps(proof))


def check_output(got, gold, denom, context, phase, profile):
    try:
        return compare(got, dict(golden=gold, denom=denom))
    except ValueError as error:
        bits = np.asarray(got, dtype="<f2").view("<u2")
        nonfinite = ~np.isfinite(got)
        bad = nonfinite | (np.abs(got.astype("f8") - gold) >= 0.005 * np.maximum(denom, 1e-30))
        indices = np.flatnonzero(bad)
        first = None
        if indices.size:
            i = int(indices[0])
            first = dict(index=i, coord=list(map(int, np.unravel_index(i, got.shape))),
                         got_bits=f"0x{bits.flat[i]:04x}", want=float(gold.flat[i]))
        proof = dict(context, phase=phase, profile=profile, error=str(error),
                     cells=got.size, bad=int(bad.sum()), nonfinite=int(nonfinite.sum()),
                     output_poison=int(np.count_nonzero(bits == 0xA5A5)),
                     metadata_poison=int(np.count_nonzero(bits == 0x7E7E)), first=first,
                     output_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                     golden_sha256=hashlib.sha256(gold.tobytes()).hexdigest())
        print("KPACK_NATIVE_OUTPUT_FAILURE " + json.dumps(proof), flush=True)
        raise OutputFailure(proof) from error


def poison_call(r, output, output_bytes, scale=None, zero=None, metadata_bytes=0):
    # All fills precede prepass/GEMM on their own nonblocking stream. A
    # default-stream memset can return early and overwrite their results.
    r.fill(output, 0xA5, output_bytes)
    if scale is not None:
        r.fill(scale, 0x7E, metadata_bytes)
        r.fill(zero, 0x7E, metadata_bytes)


def check_prepass(sdk, w, scale, zero):
    # The historical timing fixture is not the canonical metadata oracle:
    # for Q3/Q6 it starts the zero channel at -0, whereas unit_group starts
    # at +0. Keep that fixture (and its calibration hashes) unchanged. Decode
    # the actual units with the same independent oracle as the GEMV gate.
    expected = metadata_oracle(w.planes["units"], w.q, w.n, w.k, w.experts)
    proof = dict(
        q=w.q,
        n=w.n,
        k=w.k,
        experts=w.experts,
        oracle="PACKED_UNIT_FP16_V1",
        units_sha256=hashlib.sha256(w.planes["units"].tobytes()).hexdigest(),
        planes=[],
    )
    for name, ptr, want in zip(("scale", "zero"), (scale, zero), expected):
        if not np.isfinite(want).all():
            raise ValueError(f"nonfinite packed-unit oracle: q={w.q} plane={name}")
        raw = sdk.download(ptr, want.nbytes)
        if len(raw) != want.nbytes:
            raise ValueError(f"prepass readback size differs: q={w.q} plane={name}")
        got = np.frombuffer(raw, dtype="<f2").reshape(want.shape)
        got_bits, want_bits = got.view("<u2"), want.view("<u2")
        bad = got_bits != want_bits
        indices = np.flatnonzero(bad)
        first = None
        if indices.size:
            index = int(indices[0])
            expert, group, col = map(int, np.unravel_index(index, want.shape))
            first = dict(
                index=index,
                expert=expert,
                group=group,
                n=col,
                want=f"0x{want_bits.flat[index]:04x}",
                got=f"0x{got_bits.flat[index]:04x}",
            )
        proof["planes"].append(
            dict(
                plane=name,
                cells=want.size,
                bad=int(indices.size),
                signed_zero_bad=int(np.count_nonzero(bad & (got == 0) & (want == 0))),
                nonfinite=int(np.count_nonzero(~np.isfinite(got))),
                first=first,
            )
        )
    proof["status"] = "FAIL" if any(p["bad"] for p in proof["planes"]) else "PASS"
    print("KPACK_NATIVE_METADATA " + json.dumps(proof), flush=True)
    if proof["status"] != "PASS":
        raise ValueError(
            "native SF prepass differs from packed-unit oracle: " + json.dumps(proof)
        )
    return proof


def run(sdk, bundle, w, route, tokens, samples, jit=None):
    grouped = route >= 2
    e = w.experts
    if grouped:
        if tokens == 128:
            vectors = profiles(e)
        else:
            rows = np.zeros(e, dtype="i4")
            rows[np.arange(8) * 17] = 1
            vectors = [tuple(map(int, np.roll(rows, s))) for s in (0, 3, 17)]
    else:
        vectors = [(tokens,)] * 3
    m = sum(vectors[0])
    arr = arrangement(w.q)
    d = Dispatch(bundle, jit=jit)
    r = Resources(sdk)
    graph, instance = C.c_void_p(), C.c_void_p()
    try:
        choice = d.query(w.q, route, m, w.n, w.k, e, tokens, arr.mapping_id)
        if choice is None:
            raise ValueError("required native context missed its selected module")
        again = d.query(w.q, route, m, w.n, w.k, e, tokens, arr.mapping_id)
        if bytes(choice) != bytes(again):
            raise ValueError("cached selection changed")
        context = dict(q=w.q, route=route, m=m, n=w.n, k=w.k, experts=e,
                       max_rows=tokens, selection=receipt(choice))
        print("KPACK_NATIVE_SELECTED " + json.dumps(context), flush=True)
        planes = {
            x: r.upload(w.planes[x]) if w.planes[x].size else None
            for x in ("low", "high", "units")
        }
        # Pageable default-stream uploads need an explicit edge to the
        # nonblocking consumer. This is fixture setup, outside timed work.
        sdk.synchronize(None)
        sf = route in (1, 3)
        prepass_samples = []
        prepass_proof = None
        scale = zero = None
        size = 0
        if sf:
            _, _, prepare = bind(
                C.CDLL(str(bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL)
            )
            size = e * (w.k // arr.group_size) * w.n * 2
            scale, zero = r.alloc(size), r.alloc(size)
            # A skipped/partial store must not inherit valid allocator bytes.
            r.fill(scale, 0x7E, size)
            r.fill(zero, 0x7E, size)
            prepass = lambda: prepare(
                w.q,
                w.n,
                w.k,
                e,
                planes["units"],
                w.planes["units"].nbytes,
                scale,
                zero,
                size,
                C.byref(arr),
                r.stream,
            )
            prepass_samples = r.samples(prepass, 1)
            prepass_proof = check_prepass(sdk, w, scale, zero)
            prepass_samples += r.samples(prepass, 2)
        ap = r.alloc(m * w.k * 2)
        output = r.alloc(m * w.n * 2 + 32)
        bounds = r.alloc((e + 1) * 4) if grouped else None
        c = Call(
            version=1,
            size=C.sizeof(Call),
            m=m,
            n=w.n,
            k=w.k,
            experts=e,
            group_size=arr.group_size,
            device=choice.device,
            compute_units=choice.compute_units,
            mapping_id=arr.mapping_id,
            a=ap,
            low=planes["low"],
            high=planes["high"],
            metadata=scale if sf else planes["units"],
            zero=zero,
            output=output + 16,
            offsets_device=bounds,
            workspace=r.alloc(max(1, choice.workspace_bytes)),
            workspace_bytes=choice.workspace_bytes,
            stream=r.stream.value,
        )
        gemm = d.prepare(choice, c)
        def launch():
            if sf:
                checked(prepass(), "per-call SF prepass")
            return gemm()
        def check(phase, profile):
            raw = sdk.download(output, m * w.n * 2 + 32)
            if raw[:16] != b"\xa5" * 16 or raw[-16:] != b"\xa5" * 16:
                raise ValueError(f"selected output guard changed phase={phase} profile={profile}")
            got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(m, w.n)
            return check_output(got, gold, denom, context, phase, profile)
        records = []
        for index, rows in enumerate(vectors):
            req = Request(
                ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")[route],
                w.q,
                w.n,
                w.k,
                m,
                rows if grouped else (),
            )
            a, gold, denom = w.activation(req)
            checked(
                sdk.lib.hggcMemcpy(ap, a.ctypes.data, a.nbytes, 1), "A fixture upload"
            )
            if bounds:
                offsets = np.r_[0, np.cumsum(rows)].astype("i4")
                checked(
                    sdk.lib.hggcMemcpy(bounds, offsets.ctypes.data, offsets.nbytes, 1),
                    "GPU router fixture upload",
                )
            sdk.synchronize(None)
            if index == 0:
                poison_call(r, output, m * w.n * 2 + 32, scale, zero, size)
                checked(launch(), "selected eager run")
                sdk.synchronize(r.stream)
                check("eager", index)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream, 0), "begin capture")
                checked(launch(), "selected capture")
                checked(
                    sdk.lib.hggcStreamEndCapture(r.stream, C.byref(graph)),
                    "end capture",
                )
                checked(
                    sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance), graph, 0),
                    "instantiate",
                )
            # Eager success must not mask a missing graph store/prepass.
            poison_call(r, output, m * w.n * 2 + 32, scale, zero, size)
            checked(sdk.lib.hggcGraphLaunch(instance, r.stream), "selected replay")
            sdk.synchronize(r.stream)
            err = check("graph_replay", index)
            # Finite arithmetic corruption must be visible to this oracle.
            if np.max(np.abs(gold) / np.maximum(denom, 1e-30)) <= 0.005:
                raise ValueError("zero-output negative is not discriminating")
            values = r.samples(launch, samples)
            records.append(
                dict(profile=index, rows=list(rows), error=err, samples_us=values)
            )
        result = dict(
            status="PASS",
            q=w.q,
            route=route,
            m=m,
            n=w.n,
            k=w.k,
            experts=e,
            max_rows=tokens,
            selection=receipt(choice),
            profiles=records,
            prepass_samples_us=prepass_samples,
            prepass_oracle=prepass_proof,
            graph_replays=3,
            rows_host=False,
            rows_device=False,
            sf_metadata_mode="PER_CALL_GPU_PREPASS" if sf else "PACKED_UNITS",
            fixture_order="CONSUMER_STREAM_V1",
            correctness_scope="EAGER_AND_THREE_GRAPH_REPLAYS",
            scope="SELECTED_CALL_INCLUDING_PREPASS_AND_GPU_DIRECTORY_NOT_LLAMA_ADAPTERS",
        )
        print(
            "KPACK_NATIVE_CONTEXT "
            + json.dumps({k: v for k, v in result.items() if k != "profiles"}),
            flush=True,
        )
        return result
    finally:
        sdk.synchronize(r.stream)
        if instance:
            sdk.lib.hggcGraphExecDestroy(instance)
        if graph:
            sdk.lib.hggcGraphDestroy(graph)
        d.close()
        r.close()
