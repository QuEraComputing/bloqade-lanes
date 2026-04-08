#!/usr/bin/env python3
"""Visualize an ArchSpec: site positions, word groupings, buses, and entangling pairs."""

import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from bloqade.lanes.layout.encoding import LocationAddress


def visualize_arch(arch_spec, title="ArchSpec", show_buses=True, show_lanes=True):
    """Plot the architecture showing sites, words, buses, and entangling pairs.

    Args:
        arch_spec: Python ArchSpec wrapper
        title: Plot title
        show_buses: Whether to draw bus connections
        show_lanes: Whether to annotate lane addresses
    """
    inner = arch_spec._inner
    num_words = len(inner.words)
    sites_per_word = inner.sites_per_word
    num_zones = len(inner.zones)

    fig, axes = plt.subplots(1, num_zones, figsize=(14 * num_zones, 10), squeeze=False)

    for zone_id in range(num_zones):
        ax = axes[0, zone_id]
        zone = inner.zones[zone_id]

        # Collect all site positions for this zone
        positions = {}  # (word_id, site_id) -> (x, y)
        for word_id in range(num_words):
            for site_id in range(sites_per_word):
                loc = LocationAddress(word_id, site_id, zone_id)
                pos = inner.location_position(loc._inner)
                if pos is not None:
                    positions[(word_id, site_id)] = pos

        if not positions:
            ax.set_title(f"Zone {zone_id} (empty)")
            continue

        # Color words differently
        cmap = plt.cm.tab20
        word_colors = {w: cmap(w % 20) for w in range(num_words)}

        # Plot sites
        for (word_id, site_id), (x, y) in positions.items():
            color = word_colors[word_id]
            ax.plot(x, y, 'o', color=color, markersize=10, zorder=5)
            ax.annotate(
                f"w{word_id}s{site_id}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=6,
                color=color,
            )

        # Draw entangling pairs
        if zone.entangling_pairs:
            for w_a, w_b in zone.entangling_pairs:
                for site_id in range(sites_per_word):
                    pos_a = positions.get((w_a, site_id))
                    pos_b = positions.get((w_b, site_id))
                    if pos_a and pos_b:
                        ax.plot(
                            [pos_a[0], pos_b[0]],
                            [pos_a[1], pos_b[1]],
                            'r-', alpha=0.3, linewidth=2, zorder=1,
                        )

        # Draw site buses
        if show_buses and zone.site_buses:
            for bus_idx, bus in enumerate(zone.site_buses):
                for s_src, s_dst in zip(bus.src, bus.dst):
                    for word_id in zone.words_with_site_buses:
                        pos_src = positions.get((word_id, s_src))
                        pos_dst = positions.get((word_id, s_dst))
                        if pos_src and pos_dst:
                            ax.annotate(
                                '',
                                xy=pos_dst, xytext=pos_src,
                                arrowprops=dict(
                                    arrowstyle='->', color='blue',
                                    alpha=0.2, lw=1,
                                ),
                                zorder=2,
                            )

        # Draw word buses
        if show_buses and zone.word_buses:
            for bus_idx, bus in enumerate(zone.word_buses):
                for w_src, w_dst in zip(bus.src, bus.dst):
                    for site_id in range(sites_per_word):
                        pos_src = positions.get((w_src, site_id))
                        pos_dst = positions.get((w_dst, site_id))
                        if pos_src and pos_dst:
                            ax.annotate(
                                '',
                                xy=pos_dst, xytext=pos_src,
                                arrowprops=dict(
                                    arrowstyle='->', color='green',
                                    alpha=0.1, lw=0.5,
                                ),
                                zorder=2,
                            )

        # Labels
        ax.set_title(
            f"Zone {zone_id}: {num_words} words x {sites_per_word} sites | "
            f"{len(zone.site_buses)} site buses, {len(zone.word_buses)} word buses, "
            f"{len(zone.entangling_pairs)} entangling pairs"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        # Legend
        legend_items = [
            mpatches.Patch(color='red', alpha=0.3, label='Entangling pair'),
        ]
        if show_buses:
            legend_items.append(mpatches.Patch(color='blue', alpha=0.3, label='Site bus'))
            legend_items.append(mpatches.Patch(color='green', alpha=0.3, label='Word bus'))
        ax.legend(handles=legend_items, loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    spec_name = sys.argv[1] if len(sys.argv) > 1 else "logical"

    if spec_name == "logical":
        from bloqade.lanes.arch.gemini.logical.spec import get_arch_spec
        arch = get_arch_spec()
        fig = visualize_arch(arch, title="Gemini Logical ArchSpec")
    elif spec_name == "physical":
        from bloqade.lanes.arch.gemini.physical.spec import get_arch_spec
        arch = get_arch_spec()
        fig = visualize_arch(arch, title="Gemini Physical ArchSpec", show_buses=True)
    else:
        print(f"Unknown spec: {spec_name}. Use 'logical' or 'physical'.")
        sys.exit(1)

    # Print summary
    inner = arch._inner
    print(f"Words: {len(inner.words)}")
    print(f"Sites per word: {inner.sites_per_word}")
    print(f"Zones: {len(inner.zones)}")
    for i, z in enumerate(inner.zones):
        print(f"  Zone {i}: site_buses={len(z.site_buses)}, word_buses={len(z.word_buses)}, entangling_pairs={len(z.entangling_pairs)}")

    plt.savefig(f"arch_{spec_name}.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved to arch_{spec_name}.png")
    plt.show()
