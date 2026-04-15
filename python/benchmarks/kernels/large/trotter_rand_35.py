"""Randomized Trotter benchmark kernel."""

from kirin.dialects import ilist

from bloqade import squin


@squin.kernel(typeinfer=True, fold=True)
def trotter_rand_35():
    """35-qubit randomized Trotter circuit with many CZ interactions."""
    q = squin.qalloc(35)
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[4], q[33]])
    )
    squin.cz(q[4], q[33])
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[4], q[33]])
    )
    squin.cz(q[4], q[33])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[28])
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[33])
    squin.cz(q[33], q[28])
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[28], q[33]])
    )
    squin.cz(q[33], q[28])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[10])
    squin.broadcast.u3(
        1.57079632679, 0.0628318530718, -1.57079632679, ilist.IList([q[4], q[33]])
    )
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[28])
    squin.broadcast.cz(
        ilist.IList([q[28], q[33], q[4]]), ilist.IList([q[10], q[6], q[17]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[10], q[28], q[33], q[4]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[28], q[33], q[4]]), ilist.IList([q[10], q[6], q[17]])
    )
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[1], q[6], q[17]])
    )
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[10])
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[28])
    squin.broadcast.cz(
        ilist.IList([q[10], q[28], q[17]]), ilist.IList([q[1], q[8], q[6]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[1], q[10], q[28], q[6], q[17]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[10], q[28], q[17]]), ilist.IList([q[1], q[8], q[6]])
    )
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[9], q[8]])
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[1], q[6]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[10], q[4], q[33]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[1], q[10], q[6], q[4]]), ilist.IList([q[9], q[24], q[8], q[33]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[9], q[1], q[10], q[8], q[6], q[4], q[33]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[1], q[10], q[6], q[4]]), ilist.IList([q[9], q[24], q[8], q[33]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[1], q[17], q[6], q[28]]),
    )
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[24])
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[8], q[33]])
    )
    squin.broadcast.cz(
        ilist.IList([q[1], q[8], q[6], q[17], q[33]]),
        ilist.IList([q[0], q[24], q[19], q[11], q[28]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[1], q[24], q[8], q[6], q[17], q[28], q[33]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[1], q[8], q[6], q[17], q[33]]),
        ilist.IList([q[0], q[24], q[19], q[11], q[28]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[0], q[19], q[11], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[24], q[28]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[8], q[10], q[4], q[33]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[24], q[8], q[11], q[28], q[33], q[4]]),
        ilist.IList([q[0], q[23], q[19], q[10], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[24], q[0], q[8], q[19], q[11], q[10], q[28], q[33], q[4]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[24], q[8], q[11], q[28], q[33], q[4]]),
        ilist.IList([q[0], q[23], q[19], q[10], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[12], q[23], q[8], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[0], q[19], q[10]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[24], q[1], q[28]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[0], q[24], q[19], q[10], q[28], q[17]]),
        ilist.IList([q[12], q[22], q[23], q[1], q[8], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [q[12], q[0], q[24], q[23], q[19], q[1], q[10], q[28], q[17], q[6]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679, 0.0628318530718, -1.57079632679, ilist.IList([q[4], q[33]])
    )
    squin.broadcast.cz(
        ilist.IList([q[0], q[24], q[19], q[10], q[28], q[17]]),
        ilist.IList([q[12], q[22], q[23], q[1], q[8], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[0], q[10], q[11], q[19]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[9], q[23], q[1], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[22], q[24], q[8]]),
    )
    squin.cz(q[4], q[33])
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[4], q[33]])
    )
    squin.broadcast.cz(
        ilist.IList([q[0], q[1], q[23], q[10], q[19], q[11], q[6]]),
        ilist.IList([q[31], q[9], q[22], q[24], q[29], q[7], q[8]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[0], q[9], q[1], q[22], q[23], q[10], q[19], q[11], q[8], q[6]]),
    )
    squin.cz(q[4], q[33])
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[28])
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[33])
    squin.broadcast.cz(
        ilist.IList([q[0], q[1], q[23], q[10], q[19], q[11], q[6]]),
        ilist.IList([q[31], q[9], q[22], q[24], q[29], q[7], q[8]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[31], q[0], q[24], q[29], q[7], q[19], q[11]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[22], q[8]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[1], q[23], q[17], q[6]]),
    )
    squin.cz(q[33], q[28])
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[28], q[33]])
    )
    squin.broadcast.cz(
        ilist.IList([q[22], q[1], q[23], q[8], q[7], q[6], q[17]]),
        ilist.IList([q[31], q[0], q[15], q[24], q[29], q[19], q[11]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[31], q[22], q[1], q[23], q[24], q[8], q[29], q[7], q[6], q[17]]),
    )
    squin.cz(q[33], q[28])
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[33], q[28]])
    )
    squin.broadcast.cz(
        ilist.IList([q[22], q[1], q[23], q[8], q[7], q[6], q[17]]),
        ilist.IList([q[31], q[0], q[15], q[24], q[29], q[19], q[11]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[21], q[0], q[15], q[23], q[19], q[11], q[6], q[33], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[31], q[24], q[29]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[22], q[8], q[10], q[4]]),
    )
    squin.u3(1.57079632679, 0.0942477796077, -1.57079632679, q[28])
    squin.broadcast.cz(
        ilist.IList([q[31], q[22], q[24], q[29], q[8], q[28], q[11], q[33], q[4]]),
        ilist.IList([q[21], q[2], q[0], q[15], q[23], q[10], q[19], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[21],
                q[31],
                q[22],
                q[24],
                q[0],
                q[15],
                q[29],
                q[8],
                q[10],
                q[28],
                q[19],
                q[11],
                q[33],
                q[4],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[31], q[22], q[24], q[29], q[8], q[28], q[11], q[33], q[4]]),
        ilist.IList([q[21], q[2], q[0], q[15], q[23], q[10], q[19], q[6], q[17]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[12], q[0], q[15], q[10], q[19]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[31], q[7], q[24], q[29], q[1], q[28]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[2], q[22], q[23], q[8], q[6], q[17]]),
    )
    squin.broadcast.cz(
        ilist.IList(
            [q[31], q[0], q[15], q[24], q[29], q[7], q[10], q[19], q[28], q[17]]
        ),
        ilist.IList([q[20], q[12], q[2], q[22], q[14], q[30], q[1], q[23], q[8], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[31],
                q[12],
                q[0],
                q[2],
                q[15],
                q[24],
                q[29],
                q[7],
                q[1],
                q[10],
                q[23],
                q[19],
                q[28],
                q[17],
                q[6],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList(
            [q[31], q[0], q[15], q[24], q[29], q[7], q[10], q[19], q[28], q[17]]
        ),
        ilist.IList([q[20], q[12], q[2], q[22], q[14], q[30], q[1], q[23], q[8], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[9], q[2], q[1], q[23], q[6]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[20], q[31], q[22], q[24], q[14], q[30], q[29], q[7], q[8]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[0], q[15], q[11], q[10], q[19]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[2], q[0], q[15], q[1], q[23], q[10], q[30], q[19], q[11], q[6]]),
        ilist.IList(
            [q[20], q[31], q[26], q[9], q[22], q[24], q[14], q[29], q[7], q[8]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[20],
                q[2],
                q[0],
                q[15],
                q[9],
                q[1],
                q[22],
                q[23],
                q[10],
                q[30],
                q[14],
                q[19],
                q[11],
                q[8],
                q[6],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[2], q[0], q[15], q[1], q[23], q[10], q[30], q[19], q[11], q[6]]),
        ilist.IList(
            [q[20], q[31], q[26], q[9], q[22], q[24], q[14], q[29], q[7], q[8]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [q[27], q[31], q[0], q[26], q[15], q[24], q[29], q[7], q[19], q[11]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[20], q[22], q[14], q[8]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[2], q[17], q[1], q[23], q[6]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[20], q[2], q[22], q[1], q[14], q[23], q[8], q[7], q[6], q[17]]),
        ilist.IList(
            [q[27], q[3], q[31], q[0], q[26], q[15], q[24], q[29], q[19], q[11]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[27],
                q[20],
                q[2],
                q[31],
                q[22],
                q[1],
                q[26],
                q[14],
                q[23],
                q[24],
                q[8],
                q[29],
                q[7],
                q[6],
                q[17],
            ]
        ),
    )
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[30])
    squin.broadcast.cz(
        ilist.IList(
            [q[20], q[2], q[22], q[1], q[14], q[23], q[8], q[7], q[6], q[17], q[30]]
        ),
        ilist.IList(
            [q[27], q[3], q[31], q[0], q[26], q[15], q[24], q[29], q[19], q[11], q[13]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[21], q[31], q[26], q[24], q[29]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[20], q[22], q[14], q[8]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[3], q[2], q[0], q[15], q[23], q[19], q[11], q[30]]),
    )
    squin.broadcast.cz(
        ilist.IList(
            [q[20], q[31], q[26], q[22], q[14], q[24], q[29], q[8], q[11], q[30]]
        ),
        ilist.IList(
            [q[32], q[21], q[3], q[2], q[25], q[0], q[15], q[23], q[19], q[13]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[4],
                q[20],
                q[21],
                q[31],
                q[3],
                q[26],
                q[22],
                q[14],
                q[0],
                q[24],
                q[15],
                q[29],
                q[8],
                q[11],
                q[19],
                q[33],
                q[30],
                q[28],
            ]
        ),
    )
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[7])
    squin.broadcast.cz(
        ilist.IList(
            [q[20], q[31], q[26], q[22], q[14], q[24], q[29], q[8], q[11], q[7]]
        ),
        ilist.IList(
            [q[32], q[21], q[3], q[2], q[25], q[0], q[15], q[23], q[19], q[30]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [q[17], q[32], q[20], q[2], q[22], q[14], q[23], q[7], q[10], q[6]]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[12], q[3], q[0], q[15], q[19], q[27], q[21]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[31], q[26], q[24], q[29], q[11]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[3], q[31], q[26], q[0], q[15], q[24], q[29], q[19], q[7]]),
        ilist.IList([q[32], q[20], q[16], q[12], q[2], q[22], q[14], q[23], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[1],
                q[8],
                q[3],
                q[32],
                q[31],
                q[26],
                q[12],
                q[0],
                q[2],
                q[15],
                q[24],
                q[29],
                q[19],
                q[23],
                q[7],
                q[30],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[3], q[31], q[26], q[0], q[15], q[24], q[29], q[19], q[11]]),
        ilist.IList([q[32], q[20], q[16], q[12], q[2], q[22], q[14], q[23], q[7]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[5], q[20], q[31], q[26], q[22], q[14], q[29], q[24], q[11]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[32], q[2], q[23]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[3], q[0], q[15], q[19]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[3], q[2], q[0], q[15], q[23], q[30], q[19], q[11]]),
        ilist.IList([q[5], q[18], q[20], q[31], q[26], q[22], q[14], q[29], q[7]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[5],
                q[32],
                q[3],
                q[2],
                q[20],
                q[0],
                q[15],
                q[22],
                q[23],
                q[14],
                q[30],
                q[19],
                q[7],
                q[11],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[3], q[2], q[0], q[15], q[23], q[19], q[30]]),
        ilist.IList([q[5], q[18], q[20], q[31], q[26], q[22], q[29], q[14]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[32], q[2], q[23], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[3], q[31], q[0], q[26], q[15], q[29], q[19]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[20], q[5], q[22], q[14]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[20], q[2], q[22], q[14], q[23], q[30], q[7]]),
        ilist.IList([q[34], q[27], q[3], q[31], q[26], q[15], q[13], q[29]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [
                q[32],
                q[20],
                q[27],
                q[2],
                q[22],
                q[31],
                q[14],
                q[26],
                q[23],
                q[30],
                q[29],
                q[7],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[20], q[2], q[22], q[23], q[14], q[30], q[7]]),
        ilist.IList([q[34], q[27], q[3], q[31], q[15], q[26], q[13], q[29]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[32], q[2], q[3], q[15], q[23], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[20], q[22], q[14], q[7]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.125663706144,
        -1.57079632679,
        ilist.IList([q[27], q[31], q[26], q[29]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[20], q[31], q[22], q[26], q[14], q[29], q[7]]),
        ilist.IList([q[32], q[21], q[2], q[3], q[25], q[15], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList(
            [q[20], q[31], q[21], q[22], q[3], q[26], q[14], q[15], q[29], q[7]]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[20], q[31], q[22], q[26], q[14], q[29], q[7]]),
        ilist.IList([q[32], q[21], q[2], q[3], q[25], q[15], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[20], q[32], q[2], q[22], q[14], q[7], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[31], q[26], q[29]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[3], q[15]])
    )
    squin.broadcast.cz(
        ilist.IList([q[31], q[3], q[26], q[15], q[29]]),
        ilist.IList([q[20], q[32], q[16], q[2], q[14]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[31], q[3], q[32], q[26], q[2], q[15], q[29]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[31], q[3], q[26], q[15], q[29]]),
        ilist.IList([q[20], q[32], q[16], q[2], q[14]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[32], q[2]])
    )
    squin.broadcast.u3(
        1.57079632679, 0.0628318530718, -1.57079632679, ilist.IList([q[3], q[15]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[20], q[31], q[26], q[14], q[29]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[3], q[2], q[15], q[30]]),
        ilist.IList([q[5], q[18], q[20], q[26], q[14]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[32], q[5], q[3], q[2], q[20], q[15], q[14], q[30]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[3], q[2], q[15], q[30]]),
        ilist.IList([q[5], q[18], q[20], q[26], q[14]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0628318530718,
        -1.57079632679,
        ilist.IList([q[32], q[2], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.125663706144, -1.57079632679, ilist.IList([q[5], q[20], q[14]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[3], q[26], q[15]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[20], q[2], q[14], q[30]]),
        ilist.IList([q[34], q[27], q[3], q[26], q[13]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[32], q[20], q[27], q[2], q[14], q[26], q[30]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[32], q[20], q[2], q[14], q[30]]),
        ilist.IList([q[34], q[27], q[3], q[26], q[13]]),
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[32], q[3], q[2], q[30]]),
    )
    squin.broadcast.u3(
        1.57079632679, 0.0628318530718, -1.57079632679, ilist.IList([q[20], q[14]])
    )
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[26])
    squin.broadcast.cz(
        ilist.IList([q[20], q[26], q[14]]), ilist.IList([q[32], q[3], q[25]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[20], q[3], q[26], q[14]]),
    )
    squin.broadcast.cz(
        ilist.IList([q[20], q[26], q[14]]), ilist.IList([q[32], q[3], q[25]])
    )
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[32], q[20], q[14]]),
    )
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[3])
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[26])
    squin.broadcast.cz(ilist.IList([q[3], q[26]]), ilist.IList([q[32], q[16]]))
    squin.broadcast.u3(
        1.57079632679,
        0.0314159265359,
        -1.57079632679,
        ilist.IList([q[3], q[32], q[26]]),
    )
    squin.broadcast.cz(ilist.IList([q[3], q[26]]), ilist.IList([q[32], q[16]]))
    squin.u3(1.57079632679, 0.125663706144, -1.57079632679, q[32])
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[3])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[26])
    squin.broadcast.cz(ilist.IList([q[32], q[3]]), ilist.IList([q[5], q[18]]))
    squin.broadcast.u3(
        1.57079632679, 0.0314159265359, -1.57079632679, ilist.IList([q[5], q[32], q[3]])
    )
    squin.broadcast.cz(ilist.IList([q[32], q[3]]), ilist.IList([q[5], q[18]]))
    squin.broadcast.u3(
        1.57079632679,
        0.0942477796077,
        -1.57079632679,
        ilist.IList([q[9], q[12], q[21], q[27], q[5]]),
    )
    squin.u3(1.57079632679, 0.0628318530718, -1.57079632679, q[32])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[3])
    squin.cz(q[32], q[34])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[32])
    squin.cz(q[32], q[34])
    squin.u3(1.57079632679, 0.0314159265359, -1.57079632679, q[32])
