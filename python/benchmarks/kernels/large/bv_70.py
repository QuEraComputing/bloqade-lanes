from kirin.dialects import ilist

from bloqade import squin


@squin.kernel(typeinfer=True, fold=True)
def bv_70():
    q = squin.qalloc(70)
    squin.broadcast.u3(
        0,
        0,
        0,
        ilist.IList(
            [
                q[0],
                q[2],
                q[5],
                q[6],
                q[8],
                q[9],
                q[11],
                q[13],
                q[16],
                q[17],
                q[18],
                q[20],
                q[22],
                q[23],
                q[28],
                q[29],
                q[30],
                q[31],
                q[34],
                q[36],
                q[37],
                q[41],
                q[43],
                q[44],
                q[45],
                q[46],
                q[48],
                q[54],
                q[56],
                q[61],
                q[62],
                q[63],
                q[64],
            ]
        ),
    )
    squin.broadcast.u3(
        1.57079632679,
        3.14159265359,
        3.14159265359,
        ilist.IList(
            [
                q[1],
                q[3],
                q[4],
                q[7],
                q[10],
                q[12],
                q[14],
                q[15],
                q[19],
                q[21],
                q[24],
                q[25],
                q[26],
                q[27],
                q[32],
                q[33],
                q[35],
                q[38],
                q[39],
                q[40],
                q[42],
                q[47],
                q[49],
                q[50],
                q[51],
                q[52],
                q[53],
                q[55],
                q[57],
                q[58],
                q[59],
                q[60],
                q[65],
                q[67],
                q[68],
                q[69],
            ]
        ),
    )
    squin.cz(q[1], q[66])
    squin.u3(1.57079632679, 0, 3.14159265359, q[1])
    squin.u3(3.14159265359, 0, 0, q[66])
    squin.cz(q[60], q[66])
    squin.cz(q[65], q[66])
    squin.cz(q[59], q[66])
    squin.cz(q[58], q[66])
    squin.cz(q[57], q[66])
    squin.cz(q[55], q[66])
    squin.cz(q[53], q[66])
    squin.cz(q[52], q[66])
    squin.cz(q[51], q[66])
    squin.cz(q[50], q[66])
    squin.cz(q[49], q[66])
    squin.cz(q[47], q[66])
    squin.cz(q[42], q[66])
    squin.cz(q[40], q[66])
    squin.cz(q[39], q[66])
    squin.cz(q[38], q[66])
    squin.cz(q[35], q[66])
    squin.cz(q[33], q[66])
    squin.cz(q[32], q[66])
    squin.cz(q[27], q[66])
    squin.cz(q[26], q[66])
    squin.cz(q[25], q[66])
    squin.cz(q[24], q[66])
    squin.cz(q[21], q[66])
    squin.cz(q[19], q[66])
    squin.cz(q[15], q[66])
    squin.cz(q[14], q[66])
    squin.cz(q[10], q[66])
    squin.cz(q[7], q[66])
    squin.cz(q[4], q[66])
    squin.cz(q[3], q[66])
    squin.cz(q[66], q[69])
    squin.cz(q[66], q[68])
    squin.cz(q[66], q[67])
    squin.cz(q[12], q[66])
    squin.broadcast.u3(
        1.57079632679,
        -7.21644966006e-16,
        7.22267294539e-16,
        ilist.IList(
            [
                q[3],
                q[4],
                q[7],
                q[10],
                q[12],
                q[14],
                q[15],
                q[19],
                q[21],
                q[24],
                q[25],
                q[26],
                q[27],
                q[32],
                q[33],
                q[35],
                q[38],
                q[39],
                q[40],
                q[42],
                q[47],
                q[49],
                q[50],
                q[51],
                q[52],
                q[53],
                q[55],
                q[57],
                q[58],
                q[59],
                q[60],
                q[65],
                q[67],
                q[68],
                q[69],
            ]
        ),
    )
    squin.u3(1.57079632679, 0, 6.28318530718, q[66])
