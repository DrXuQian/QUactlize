"""Independent correctness and rotating-weight ordinary cuBLAS calls."""
import time

from dev.strategy.prefill_measure import expect_oracle_failure
from dev.strategy.cublas_timing import NativeCalls, FRONTEND
from quactlize.runtime.native import checked
from tools.kpack_bf16_providers import loaded_images


PHASES = ('input', 'provider', 'output')


def measure(b, core, *, config=4):
    if b.w.operation != 'dense' or b.w.experts != 1:
        raise ValueError('ordinary cuBLAS worker requires a dense workload')
    start = time.monotonic()
    submit = NativeCalls(b, core)
    def launch(copy=0):
        checked(b.input(core.a.data_ptr(), True), 'ordinary BF16 input')
        checked(submit.matmul(copy), 'ordinary native BF16 GEMM')
        return b.finish(core.out.data_ptr(), True)
    numerical = []
    for repeat in (0, 1):
        b.update(repeat); core.setup(config)
        for copy in (0, core.copies-1):
            checked(launch(copy), 'ordinary cuBLAS independent oracle')
            numerical.append(b.check())
    # This provider has BF16 inputs regardless of the quantized TC control's
    # compute type. Preserve the BF16 large-activation control as well.
    b.update(2, amplitude=2**20); core.setup(config)
    checked(launch(), 'ordinary BF16 large activation'); numerical.append(b.check())
    b.update(0, zero=True); core.setup(config)
    checked(launch(), 'ordinary zero A'); b.check(zero=True)
    expect_oracle_failure(b.check, 'ordinary zero-output negative')
    b.update(0); core.setup(config)
    back = core.weights[-1]
    try:
        b.r.fill(back.data_ptr(), 0, back.numel()*2); b.rt.sync()
        checked(launch(core.copies-1), 'ordinary last-copy negative')
        expect_oracle_failure(b.check, 'ordinary last-copy BF16 weights')
        checked(launch(0), 'ordinary copy0 isolation'); b.check()
    finally:
        core.setup(config)
    weight_proof = [core.check_weights(i) for i in (0, core.copies-1)]
    # Poison the intermediate provider output as well as the final output:
    # merely running the finish kernel must not satisfy the GEMM oracle.
    b.update(1); core.update()
    b.r.fill(core.out.data_ptr(), 0, core.out.numel()*2)
    checked(b.input(core.a.data_ptr(), True), 'omitted producer input')
    checked(b.finish(core.out.data_ptr(), True), 'omitted producer finish')
    expect_oracle_failure(b.check, 'ordinary omitted producer')
    for i in range(core.copies): checked(launch(i), 'ordinary changed-input ring')
    numerical.append(b.check())
    b.update(0); core.update()
    checked(launch(), 'ordinary original-input oracle'); numerical.append(b.check())
    # All CPU checking/image hashing finishes before the native warmup. No
    # D2H, Python stream/view setup, logging, or per-sample synchronization
    # is interleaved with the timed rounds.
    warmups = max(2, (11+core.copies-1)//core.copies)*core.copies
    images = loaded_images()
    if not images: raise ValueError('installed cuBLAS images missing after warmup')
    measured = submit.measure(warmups)
    count = max(3, core.copies)
    numerical.append(b.check())
    print('STRATEGY_CUBLAS_PROGRESS mode=native-events '
          f'rounds=3/3 ring={core.copies} cuda_graph=0', flush=True)
    checked(launch(), 'ordinary final oracle'); numerical.append(b.check())
    core.guards()
    for buffer in b.ring.buffers: buffer.guard()
    if images != loaded_images(): raise ValueError('cuBLAS images changed during measurement')
    return dict(correctness='PASS', numerical=numerical, weights=weight_proof,
        pointer_negatives=['BF16_LAST_COPY_ZERO'], zero_a='PASS', omitted_producer='DETECTED',
        changed_input='PASS', receipt=core.receipt, fixture=b.ring.receipt, measured=measured,
        warmup_calls=warmups, event_warmup_calls=1, rounds=3, samples_per_round=count, cuda_graph=False,
        provider_loaded_images=images, annotations=None,
        submission_frontend=FRONTEND, per_sample_synchronizations=0, inter_round_checks=0,
        event_scope='SEPARATE_API_EVENTS_PROVIDER_INCLUDES_INTERNAL_REDUCTION_AND_LAUNCH_GAPS',
        profiler=False, wall_seconds=time.monotonic()-start,
        production_changed=False, performance_admitted=False)
