from kirin.dialects import ilist

from bloqade import squin


@squin.kernel(typeinfer=True, fold=True)
def bv_70():
    q = squin.qalloc(70)
    squin.broadcast.u3(
        1.57079632679,
        3.14159265359,
        3.14159265359,
        ilist.IList(
            [
                q[12],
                q[67],
                q[68],
                q[69],
                q[3],
                q[4],
                q[7],
                q[10],
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
                q[1],
            ]
        ),
    )
    squin.broadcast.u3(
        0,
        0,
        0,
        ilist.IList(
            [
                q[43],
                q[44],
                q[28],
                q[62],
                q[36],
                q[46],
                q[11],
                q[31],
                q[2],
                q[16],
                q[37],
                q[6],
                q[61],
                q[23],
                q[0],
                q[48],
                q[13],
                q[54],
                q[56],
                q[17],
                q[34],
                q[8],
                q[41],
                q[45],
                q[18],
                q[20],
                q[64],
                q[30],
                q[22],
                q[5],
                q[9],
                q[63],
                q[29],
            ]
        ),
    )
    squin.cz(q[1], q[66])
    squin.u3(3.14159265359, 0, 0, q[66])
    squin.u3(1.57079632679, 0, 3.14159265359, q[1])
    squin.cz(q[60], q[66])
    squin.cz(q[66], q[65])
    squin.cz(q[66], q[59])
    squin.cz(q[66], q[58])
    squin.cz(q[66], q[57])
    squin.cz(q[66], q[55])
    squin.cz(q[66], q[53])
    squin.cz(q[66], q[52])
    squin.cz(q[66], q[51])
    squin.cz(q[66], q[50])
    squin.cz(q[66], q[49])
    squin.cz(q[66], q[47])
    squin.cz(q[66], q[42])
    squin.cz(q[66], q[40])
    squin.cz(q[66], q[39])
    squin.cz(q[66], q[38])
    squin.cz(q[66], q[35])
    squin.cz(q[66], q[33])
    squin.cz(q[66], q[32])
    squin.cz(q[66], q[27])
    squin.cz(q[66], q[26])
    squin.cz(q[66], q[25])
    squin.cz(q[66], q[24])
    squin.cz(q[66], q[21])
    squin.cz(q[66], q[19])
    squin.cz(q[66], q[15])
    squin.cz(q[66], q[14])
    squin.cz(q[66], q[10])
    squin.cz(q[66], q[7])
    squin.cz(q[66], q[4])
    squin.cz(q[66], q[3])
    squin.cz(q[66], q[69])
    squin.cz(q[66], q[68])
    squin.cz(q[66], q[67])
    squin.cz(q[66], q[12])
    squin.u3(1.57079632679, 0, 6.28318530718, q[66])
    squin.broadcast.u3(
        1.57079632679,
        -7.21644966006e-16,
        7.22267294539e-16,
        ilist.IList(
            [
                q[12],
                q[67],
                q[68],
                q[69],
                q[3],
                q[4],
                q[7],
                q[10],
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
            ]
        ),
    )
