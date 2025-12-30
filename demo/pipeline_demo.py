import difflib
import io
from typing import Any

from bloqade.gemini.dialects import logical as gemini_logical
from bloqade.stim.emit import EmitStimMain
from bloqade.stim.upstream import squin_to_stim
from kirin.dialects import debug, ilist

from bloqade import qubit, squin, stim
from bloqade.lanes.logical_mvp import (
    compile_to_physical_squin_noise_model,
)
from bloqade.lanes.transform import SimpleNoiseModel

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel
def main():
    q = squin.qalloc(9)

    # random params
    theta = 0.234
    phi_0 = 0.123
    phi_1 = 0.934
    phi_2 = 0.343

    # initialization (encoding)
    squin.broadcast.rx(2 * theta, ilist.IList([q[3], q[4], q[6], q[7]]))
    squin.broadcast.h(ilist.IList([q[3], q[4], q[6], q[7]]))
    squin.rx(phi_2, q[2])
    squin.rx(phi_1, q[5])
    squin.rx(phi_0, q[8])

    # transversal logic
    squin.broadcast.cx(ilist.IList([q[2], q[3]]), ilist.IList([q[0], q[1]]))
    squin.cx(q[0], q[1])
    squin.cx(q[4], q[1])
    squin.broadcast.cx(ilist.IList([q[5], q[6]]), ilist.IList([q[0], q[1]]))
    squin.cx(q[0], q[1])
    squin.broadcast.cx(ilist.IList([q[7], q[8]]), ilist.IList([q[1], q[0]]))

    gemini_logical.terminal_measure(q)


@squin.kernel
def cz_unpaired_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    debug.info("CZ Unpaired Noise")
    squin.broadcast.depolarize(0.001, qubits)
    squin.broadcast.qubit_loss(0.0001, qubits)


@squin.kernel
def lane_noise(qubit: qubit.Qubit):
    debug.info("Lane Noise")
    squin.depolarize(0.002, qubit)
    squin.qubit_loss(0.0002, qubit)


@squin.kernel
def idle_noise(qubits: ilist.IList[qubit.Qubit, Any]):
    debug.info("Idle Noise")
    squin.broadcast.depolarize(0.0005, qubits)
    squin.broadcast.qubit_loss(0.00005, qubits)


noise_kernel = compile_to_physical_squin_noise_model(
    main,
    SimpleNoiseModel(lane_noise, idle_noise, cz_unpaired_noise),
)
noise_kernel.print()
noise_kernel = squin_to_stim(noise_kernel)

buf = io.StringIO()
emit = EmitStimMain(dialects=stim.main, io=buf)
emit.initialize()
emit.run(node=noise_kernel)
result = buf.getvalue().strip()

expected = """# Begin Steane7 Initialize
SQRT_Y_DAG 0 1 2 3 4 5
CZ 1 0 3 2 5 4
SQRT_Y 6
CZ 0 3 2 5 4 6
SQRT_Y 2 3 4 5 6
CZ 0 1 2 3 4 5
SQRT_Y 1 2 4
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 7 8 9 10 11 12
CZ 8 7 10 9 12 11
SQRT_Y 13
CZ 7 10 9 12 11 13
SQRT_Y 9 10 11 12 13
CZ 7 8 9 10 11 12
SQRT_Y 8 9 11
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 14 15 16 17 18 19
CZ 15 14 17 16 19 18
SQRT_Y 20
CZ 14 17 16 19 18 20
SQRT_Y 16 17 18 19 20
CZ 14 15 16 17 18 19
SQRT_Y 15 16 18
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 21 22 23 24 25 26
CZ 22 21 24 23 26 25
SQRT_Y 27
CZ 21 24 23 26 25 27
SQRT_Y 23 24 25 26 27
CZ 21 22 23 24 25 26
SQRT_Y 22 23 25
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 28 29 30 31 32 33
CZ 29 28 31 30 33 32
SQRT_Y 34
CZ 28 31 30 33 32 34
SQRT_Y 30 31 32 33 34
CZ 28 29 30 31 32 33
SQRT_Y 29 30 32
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 35 36 37 38 39 40
CZ 36 35 38 37 40 39
SQRT_Y 41
CZ 35 38 37 40 39 41
SQRT_Y 37 38 39 40 41
CZ 35 36 37 38 39 40
SQRT_Y 36 37 39
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 42 43 44 45 46 47
CZ 43 42 45 44 47 46
SQRT_Y 48
CZ 42 45 44 47 46 48
SQRT_Y 44 45 46 47 48
CZ 42 43 44 45 46 47
SQRT_Y 43 44 46
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 49 50 51 52 53 54
CZ 50 49 52 51 54 53
SQRT_Y 55
CZ 49 52 51 54 53 55
SQRT_Y 51 52 53 54 55
CZ 49 50 51 52 53 54
SQRT_Y 50 51 53
# End Steane7 Initialize
# Begin Steane7 Initialize
SQRT_Y_DAG 56 57 58 59 60 61
CZ 57 56 59 58 61 60
SQRT_Y 62
CZ 56 59 58 61 60 62
SQRT_Y 58 59 60 61 62
CZ 56 57 58 59 60 61
SQRT_Y 57 58 60
# End Steane7 Initialize
S 21 22 23 24 25 26 27 28 29 30 31 32 33 34 42 43 44 45 46 47 48 49 50 51 52 53 54 55
SQRT_X_DAG 21 22 23 24 25 26 27 28 29 30 31 32 33 34 42 43 44 45 46 47 48 49 50 51 52 53 54 55
S 21 22 23 24 25 26 27 28 29 30 31 32 33 34 42 43 44 45 46 47 48 49 50 51 52 53 54 55
SQRT_Y 0 1 2 3 4 5 6 7 8 9 10 11 12 13
CZ 0 14 1 15 2 16 3 17 4 18 5 19 6 20 7 21 8 22 9 23 10 24 11 25 12 26 13 27
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
I_ERROR[loss](0.00010000) 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
SQRT_Y_DAG 0 1 2 3 4 5 6 7 8 9 10 11 12 13
SQRT_Y 7 8 9 10 11 12 13
# Idle Noise
DEPOLARIZE1(0.00050000) 0 1 2 3 4 5 6
I_ERROR[loss](0.00005000) 0 1 2 3 4 5 6
CZ 7 0 8 1 9 2 10 3 11 4 12 5 13 6
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
I_ERROR[loss](0.00010000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
# Idle Noise
DEPOLARIZE1(0.00050000) 0 1 2 3 4 5 6
I_ERROR[loss](0.00005000) 0 1 2 3 4 5 6
SQRT_Y_DAG 7 8 9 10 11 12 13
SQRT_Y 7 8 9 10 11 12 13
CZ 7 28 8 29 9 30 10 31 11 32 12 33 13 34
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 0 1 2 3 4 5 6 14 15 16 17 18 19 20 21 22 23 24 25 26 27 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
I_ERROR[loss](0.00010000) 0 1 2 3 4 5 6 14 15 16 17 18 19 20 21 22 23 24 25 26 27 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
SQRT_Y_DAG 7 8 9 10 11 12 13
SQRT_Y 0 1 2 3 4 5 6 7 8 9 10 11 12 13
CZ 0 35 1 36 2 37 3 38 4 39 5 40 6 41 7 42 8 43 9 44 10 45 11 46 12 47 13 48
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 49 50 51 52 53 54 55 56 57 58 59 60 61 62
I_ERROR[loss](0.00010000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 49 50 51 52 53 54 55 56 57 58 59 60 61 62
SQRT_Y_DAG 0 1 2 3 4 5 6 7 8 9 10 11 12 13
SQRT_Y 7 8 9 10 11 12 13
CZ 0 7 1 8 2 9 3 10 4 11 5 12 6 13
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
I_ERROR[loss](0.00010000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
SQRT_Y_DAG 7 8 9 10 11 12 13
SQRT_Y 7 8 9 10 11 12 13 0 1 2 3 4 5 6
CZ 0 56 1 57 2 58 3 59 4 60 5 61 6 62 7 49 8 50 9 51 10 52 11 53 12 54 13 55
# CZ Unpaired Noise
DEPOLARIZE1(0.00100000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
I_ERROR[loss](0.00010000) 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
SQRT_Y_DAG 7 8 9 10 11 12 13 0 1 2 3 4 5 6
MZ(0.00000000) 0
MZ(0.00000000) 1
MZ(0.00000000) 2
MZ(0.00000000) 3
MZ(0.00000000) 4
MZ(0.00000000) 5
MZ(0.00000000) 6
MZ(0.00000000) 7
MZ(0.00000000) 8
MZ(0.00000000) 9
MZ(0.00000000) 10
MZ(0.00000000) 11
MZ(0.00000000) 12
MZ(0.00000000) 13
MZ(0.00000000) 14
MZ(0.00000000) 15
MZ(0.00000000) 16
MZ(0.00000000) 17
MZ(0.00000000) 18
MZ(0.00000000) 19
MZ(0.00000000) 20
MZ(0.00000000) 21
MZ(0.00000000) 22
MZ(0.00000000) 23
MZ(0.00000000) 24
MZ(0.00000000) 25
MZ(0.00000000) 26
MZ(0.00000000) 27
MZ(0.00000000) 28
MZ(0.00000000) 29
MZ(0.00000000) 30
MZ(0.00000000) 31
MZ(0.00000000) 32
MZ(0.00000000) 33
MZ(0.00000000) 34
MZ(0.00000000) 35
MZ(0.00000000) 36
MZ(0.00000000) 37
MZ(0.00000000) 38
MZ(0.00000000) 39
MZ(0.00000000) 40
MZ(0.00000000) 41
MZ(0.00000000) 42
MZ(0.00000000) 43
MZ(0.00000000) 44
MZ(0.00000000) 45
MZ(0.00000000) 46
MZ(0.00000000) 47
MZ(0.00000000) 48
MZ(0.00000000) 49
MZ(0.00000000) 50
MZ(0.00000000) 51
MZ(0.00000000) 52
MZ(0.00000000) 53
MZ(0.00000000) 54
MZ(0.00000000) 55
MZ(0.00000000) 56
MZ(0.00000000) 57
MZ(0.00000000) 58
MZ(0.00000000) 59
MZ(0.00000000) 60
MZ(0.00000000) 61
MZ(0.00000000) 62"""

try:
    assert result == expected, "Output does not match expected."
except AssertionError:
    diff = difflib.Differ().compare(
        expected.splitlines(),
        result.splitlines(),
    )
    print("\n".join(diff))
