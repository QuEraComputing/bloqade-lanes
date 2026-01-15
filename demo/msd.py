from bloqade.gemini import logical as gemini_logical
from tsim import Circuit

from bloqade import qubit, squin
from bloqade.lanes.logical_mvp import (
    Decoder,
    compile_to_physical_stim_program,
    kernel,
    set_detector,
    set_observable,
)


@kernel
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.t(reg)

    squin.broadcast.sqrt_x([reg[0], reg[1], reg[4]])
    squin.broadcast.cz([reg[0], reg[2]], [reg[1], reg[3]])
    squin.broadcast.sqrt_y([reg[0], reg[3]])
    squin.broadcast.cz([reg[0], reg[3]], [reg[2], reg[4]])
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz([reg[0], reg[1]], [reg[4], reg[3]])
    squin.broadcast.sqrt_y_adj(reg)

    measurements = gemini_logical.terminal_measure(reg)

    for i in range(len(reg)):
        set_detector(measurements[i])
        set_observable(measurements[i], i)


program_text = compile_to_physical_stim_program(main)
circuit = Circuit(program_text)
sampler = circuit.compile_detector_sampler()
detector_bits, logical_bits = sampler.sample(100, separate_observables=True)
print(logical_bits)

dem = circuit.detector_error_model(approximate_disjoint_errors=True)
decoder = Decoder(dem)
corrected_logical_bits = decoder.decode(detector_bits, logical_bits)
print(corrected_logical_bits)
