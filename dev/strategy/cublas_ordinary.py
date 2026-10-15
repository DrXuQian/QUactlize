"""Independent correctness and rotating-weight ordinary cuBLAS calls."""
import time

from dev.strategy.prefill_measure import expect_oracle_failure
from quactlize.runtime.native import checked
from tools.kpack_bf16_providers import loaded_images


PHASES = ('input', 'provider', 'output')


def measure(b, core, *, config=4):
    if b.w.operation != 'dense' or b.w.experts != 1:
        raise ValueError('ordinary cuBLAS worker requires a dense workload')
    start = time.monotonic()
    numerical = []
    for repeat in (0, 1):
        b.update(repeat); core.setup(config)
        for copy in (0, core.copies-1):
            checked(core.launch(copy), 'ordinary cuBLAS independent oracle')
            numerical.append(b.check())
    # This provider has BF16 inputs regardless of the quantized TC control's
    # compute type. Preserve the BF16 large-activation control as well.
    b.update(2, amplitude=2**20); core.setup(config)
    checked(core.launch(), 'ordinary BF16 large activation'); numerical.append(b.check())
    b.update(0, zero=True); core.setup(config)
    checked(core.launch(), 'ordinary zero A'); b.check(zero=True)
    expect_oracle_failure(b.check, 'ordinary zero-output negative')
    b.update(0); core.setup(config)
    back = core.weights[-1]
    try:
        b.r.fill(back.data_ptr(), 0, back.numel()*2); b.rt.sync()
        checked(core.launch(core.copies-1), 'ordinary last-copy negative')
        expect_oracle_failure(b.check, 'ordinary last-copy BF16 weights')
        checked(core.launch(0), 'ordinary copy0 isolation'); b.check()
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
    for i in range(core.copies): checked(core.launch(i), 'ordinary changed-input ring')
    numerical.append(b.check())
    b.update(0); core.update()
    # Standard untimed warmup, not a claim that identical dispatches prove
    # optimality. Every rotating weight address is visited twice, at least11
    # complete calls; synchronize before any measured event.
    warmups = max(2, (11+core.copies-1)//core.copies)*core.copies
    for i in range(warmups): checked(core.launch(i % core.copies), 'ordinary warmup')
    # Warm the event API too. End at the last address so sample0 does not
    # immediately reuse a just-warmed copy0 for a two-copy large-weight ring.
    b.r.samples(lambda: core.matmul(core.copies-1), 1)
    numerical.append(b.check()); b.rt.sync()
    images = loaded_images()
    if not images: raise ValueError('installed cuBLAS images missing after warmup')
    measured = []
    count = max(3, core.copies)
    for round_ in range(3):
        for sample in range(count):
            copy = (round_*count+sample) % core.copies
            for phase in PHASES:
                row = dict(round=round_, sample=sample,
                           copy=copy, phase=phase)
                def launch():
                    if phase == 'input': return b.input(core.a.data_ptr(), True)
                    if phase == 'output': return b.finish(core.out.data_ptr(), True)
                    return core.matmul(copy)
                # Each interval contains exactly one API call. In the provider
                # phase that call may launch GEMM plus an internal reducer.
                # Neither dequant nor caller input/output adapters enter it.
                row['event_span_us'] = b.r.samples(launch, 1)[0]
                measured.append(row)
        numerical.append(b.check())
        print('STRATEGY_CUBLAS_PROGRESS mode=events '
              f'round={round_+1}/3 ring={core.copies} cuda_graph=0', flush=True)
    checked(core.launch(), 'ordinary final oracle'); numerical.append(b.check())
    core.guards()
    for buffer in b.ring.buffers: buffer.guard()
    if images != loaded_images(): raise ValueError('cuBLAS images changed during measurement')
    return dict(correctness='PASS', numerical=numerical, weights=weight_proof,
        pointer_negatives=['BF16_LAST_COPY_ZERO'], zero_a='PASS', omitted_producer='DETECTED',
        changed_input='PASS', receipt=core.receipt, fixture=b.ring.receipt, measured=measured,
        warmup_calls=warmups, event_warmup_calls=1, rounds=3, samples_per_round=count, cuda_graph=False,
        provider_loaded_images=images, annotations=None,
        event_scope='SEPARATE_API_EVENTS_PROVIDER_INCLUDES_INTERNAL_REDUCTION_AND_LAUNCH_GAPS',
        profiler=False, wall_seconds=time.monotonic()-start,
        production_changed=False, performance_admitted=False)
