from unittest.mock import Mock

from kirin import ir
from matplotlib import pyplot as plt

from bloqade.lanes.analysis.atom import AtomState, AtomStateData
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import ArchSpec, LocationAddress, SiteLaneAddress
from bloqade.lanes.visualize import artist


def make_word():
    word = Mock()
    word.site_indices = [0, 1]
    word.site_position.side_effect = lambda i: (i, i)
    word.all_positions.return_value = [(0, 0), (1, 1)]
    return word


def make_arch_spec():
    arch = Mock(spec=ArchSpec)
    arch.words = (make_word(),)
    arch.x_bounds = (0, 1)
    arch.y_bounds = (0, 1)
    arch.zones = {0: [0]}
    arch.get_path.return_value = ((0.0, 0.0), (1.0, 1.0))
    arch.get_position.return_value = (0.0, 0.0)
    arch.path_bounds.return_value = (0.0, 1.0, 0.0, 1.0)
    arch.plot.return_value = None
    return arch


def make_atom_state():
    # Use a real AtomState with minimal valid data and correct types
    # Use SiteLaneAddress for LaneAddress
    lane_addr = SiteLaneAddress(word_id=0, site_id=0, bus_id=0)
    loc_addr = LocationAddress(0, 0)
    data = AtomStateData(
        prev_lanes={0: lane_addr},
        qubit_to_locations={0: loc_addr},
        locations_to_qubit={loc_addr: 0},
    )
    return AtomState(data)


def test_state_artist_move_renderer_and_draw_methods():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params)
    state = make_atom_state()
    mr = sa.move_renderer(state, speed=1.0)
    assert mr is not None
    sa.draw_atoms(state)
    sa.draw_moves(state)
    plt.close(fig)


def test_state_artist_show_methods():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params)
    # Use mocks for move.* and ir.Statement
    stmt_localr = Mock(spec=move.LocalR)
    stmt_localr.location_addresses = [Mock(word_id=0, site_id=0)]
    stmt_localrz = Mock(spec=move.LocalRz)
    stmt_localrz.location_addresses = [Mock(word_id=0, site_id=0)]
    stmt_globalr = Mock(spec=move.GlobalR)
    stmt_globalrz = Mock(spec=move.GlobalRz)
    stmt_cz = Mock(spec=move.CZ)
    stmt_cz.zone_address = Mock(zone_id=0)
    stmt_ir = Mock(spec=ir.Statement)
    sa.show_local_r(stmt_localr)
    sa.show_local_rz(stmt_localrz)
    sa.show_global_r(stmt_globalr)
    sa.show_global_rz(stmt_globalrz)
    sa.show_cz(stmt_cz)
    sa.show_slm(stmt_ir, atom_marker="o")
    plt.close(fig)


def test_get_state_artist_and_drawer():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    sa = artist.get_state_artist(arch, ax, atom_marker="o")
    assert isinstance(sa, artist.StateArtist)
    plt.close(fig)
