//! Arena-based search graph with transposition table.
//!
//! [`SearchGraph`] stores search nodes in a flat arena (`Vec<NodeData>`)
//! indexed by [`NodeId`]. A transposition table maps configurations to the
//! best-known node (lowest g-score), using the actual cost rather than depth.
//!
//! Path reconstruction walks parent pointers — no children are stored.

use std::collections::HashMap;

use crate::primitives::config::Config;

/// Opaque handle to a node in the search graph.
///
/// Internally an index into the node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct NodeId(pub(crate) u32);

// `MoveSet` was moved to `bloqade_lanes_dsl_core::primitives::move_set` so
// the dsl-core `ArchSpec` methods can return a validated `StarlarkMoveSet`
// without dsl-core needing to depend on this crate. Re-exported here so
// existing `crate::primitives::graph::MoveSet` imports across this crate keep working.
pub use bloqade_lanes_dsl_core::primitives::move_set::MoveSet;

/// Internal node storage.
struct NodeData {
    config: Config,
    parent: Option<NodeId>,
    parent_move: Option<MoveSet>,
    g_score: f64,
    depth: u32,
    /// The configuration's slot: one per distinct configuration, shared by
    /// every node that holds it (a cheaper rediscovery mints a new node on
    /// the same slot).
    slot: u32,
}

/// Arena-based search graph with transposition table.
///
/// Nodes are stored in a flat `Vec` and referenced by [`NodeId`].
/// The transposition table maps each unique configuration to a **slot**, and
/// each slot to the [`NodeId`] with the lowest known g-score for that
/// configuration. Slots are dense indices assigned in discovery order, so a
/// per-configuration memo (a bound estimate, a children cache) is a `Vec`
/// indexed by [`Self::slot`] rather than a map keyed by `Config`.
///
/// Unlike the Python `ConfigurationTree`:
/// - Uses g-score (cost) for the transposition table, not depth.
/// - Does not store children — only parent pointers for path reconstruction.
/// - Arena allocation avoids reference cycles and per-node heap allocation.
pub struct SearchGraph {
    nodes: Vec<NodeData>,
    /// Configuration → slot.
    seen: HashMap<Config, u32>,
    /// Slot → the current best node for that configuration.
    slots: Vec<NodeId>,
}

impl std::fmt::Debug for SearchGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchGraph")
            .field("num_nodes", &self.nodes.len())
            .field("num_configs", &self.slots.len())
            .finish()
    }
}

impl SearchGraph {
    /// Create a new search graph rooted at the given configuration (g = 0).
    pub fn new(root: Config) -> Self {
        let root_node = NodeData {
            config: root.clone(),
            parent: None,
            parent_move: None,
            g_score: 0.0,
            depth: 0,
            slot: 0,
        };
        let mut seen = HashMap::new();
        seen.insert(root, 0);
        Self {
            nodes: vec![root_node],
            seen,
            slots: vec![NodeId(0)],
        }
    }

    /// The root node ID.
    pub fn root(&self) -> NodeId {
        NodeId(0)
    }

    /// Get the configuration of a node.
    pub fn config(&self, id: NodeId) -> &Config {
        &self.nodes[id.0 as usize].config
    }

    /// Get the g-score (accumulated cost from root) of a node.
    pub fn g_score(&self, id: NodeId) -> f64 {
        self.nodes[id.0 as usize].g_score
    }

    /// Get the depth of a node (number of steps from root). O(1).
    pub fn depth(&self, id: NodeId) -> u32 {
        self.nodes[id.0 as usize].depth
    }

    /// Return the parent node, or `None` for the root.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[id.0 as usize].parent
    }

    /// Number of nodes in the arena (always >= 1 due to root).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Number of distinct configurations seen (always >= 1 due to root).
    ///
    /// `len() - num_configs()` is the number of superseded nodes: ids minted
    /// by a cheaper rediscovery that left an older node for the same
    /// configuration in place.
    pub fn num_configs(&self) -> usize {
        self.slots.len()
    }

    /// The slot of a node's configuration: a dense index shared by every node
    /// holding that configuration, suitable for indexing a per-configuration
    /// memo.
    pub fn slot(&self, id: NodeId) -> u32 {
        self.nodes[id.0 as usize].slot
    }

    /// Whether `id` is the best-known node for its configuration, i.e. no
    /// cheaper rediscovery has superseded it. No hashing: a slot lookup.
    pub fn is_current(&self, id: NodeId) -> bool {
        self.slots[self.nodes[id.0 as usize].slot as usize] == id
    }

    /// Always returns `false` — the graph always contains at least the root.
    ///
    /// Provided to satisfy the `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Look up the best-known [`NodeId`] for a configuration.
    pub fn seen_id(&self, config: &Config) -> Option<NodeId> {
        self.seen.get(config).map(|&slot| self.slots[slot as usize])
    }

    /// Try to insert a successor node.
    ///
    /// Returns `(node_id, true)` if a new node was created (either the
    /// config was unseen, or it was re-discovered at a lower g-score).
    ///
    /// Returns `(existing_id, false)` if the config was already seen at
    /// an equal-or-lower g-score.
    ///
    /// On cheaper re-discovery, a **new** `NodeId` is created (lazy
    /// deletion strategy). The old `NodeId` remains in the arena but
    /// the transposition table now points to the new one.
    pub fn insert(
        &mut self,
        parent: NodeId,
        move_set: MoveSet,
        new_config: Config,
        new_g: f64,
    ) -> (NodeId, bool) {
        let existing_slot = self.seen.get(&new_config).copied();
        if let Some(slot) = existing_slot {
            let existing_id = self.slots[slot as usize];
            let existing_g = self.nodes[existing_id.0 as usize].g_score;
            if existing_g <= new_g {
                // Already seen at equal-or-lower cost.
                return (existing_id, false);
            }
            // Re-discovered at lower cost: create new node on the same slot,
            // repoint the slot.
        }

        let parent_depth = self.nodes[parent.0 as usize].depth;
        let new_id =
            NodeId(u32::try_from(self.nodes.len()).expect("search graph exceeded 2^32 nodes"));
        let slot = match existing_slot {
            Some(slot) => {
                self.slots[slot as usize] = new_id;
                slot
            }
            None => {
                let slot =
                    u32::try_from(self.slots.len()).expect("search graph exceeded 2^32 configs");
                self.slots.push(new_id);
                self.seen.insert(new_config.clone(), slot);
                slot
            }
        };
        self.nodes.push(NodeData {
            config: new_config,
            parent: Some(parent),
            parent_move: Some(move_set),
            g_score: new_g,
            depth: parent_depth + 1,
            slot,
        });
        (new_id, true)
    }

    /// Reconstruct the path from root to this node.
    ///
    /// Returns the sequence of [`MoveSet`]s in root-to-node order.
    /// For the root node, returns an empty vec.
    pub fn reconstruct_path(&self, id: NodeId) -> Vec<MoveSet> {
        let mut moves = Vec::new();
        let mut current = id;
        while let Some(parent_id) = self.nodes[current.0 as usize].parent {
            let move_set = self.nodes[current.0 as usize]
                .parent_move
                .as_ref()
                .expect("non-root node must have parent_move")
                .clone();
            moves.push(move_set);
            current = parent_id;
        }
        moves.reverse();
        moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{lane, loc};

    /// One-atom config at site `site`, for building arena chains.
    fn cfg(site: u32) -> Config {
        Config::new([(0, loc(0, site))]).unwrap()
    }

    #[test]
    fn root_creation() {
        let cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let graph = SearchGraph::new(cfg.clone());

        assert_eq!(graph.len(), 1);
        assert_eq!(*graph.config(graph.root()), cfg);
        assert_eq!(graph.g_score(graph.root()), 0.0);
    }

    #[test]
    fn insert_new_config() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]).unwrap();
        let ms = MoveSet::new([lane(0, 0, 0)]);
        let (id, is_new) = graph.insert(graph.root(), ms, child_cfg, 1.0);

        assert!(is_new);
        assert_eq!(graph.g_score(id), 1.0);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn insert_same_config_higher_cost_rejected() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]).unwrap();
        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let (first_id, _) = graph.insert(graph.root(), ms1, child_cfg.clone(), 1.0);

        let ms2 = MoveSet::new([lane(0, 0, 1)]);
        let (returned_id, is_new) = graph.insert(graph.root(), ms2, child_cfg, 2.0);

        assert!(!is_new);
        assert_eq!(returned_id, first_id);
        assert_eq!(graph.len(), 2); // no new node created
    }

    #[test]
    fn insert_same_config_lower_cost_creates_new_node() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]).unwrap();
        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let (first_id, _) = graph.insert(graph.root(), ms1, child_cfg.clone(), 5.0);

        let ms2 = MoveSet::new([lane(0, 0, 1)]);
        let (second_id, is_new) = graph.insert(graph.root(), ms2, child_cfg.clone(), 2.0);

        assert!(is_new);
        assert_ne!(first_id, second_id);
        assert_eq!(graph.g_score(second_id), 2.0);
        // Transposition table now points to the cheaper node.
        assert_eq!(graph.seen_id(&child_cfg), Some(second_id));
        // Old node still accessible.
        assert_eq!(graph.g_score(first_id), 5.0);
        assert_eq!(graph.len(), 3);
    }

    /// A cheaper rediscovery mints a new node on the *same* slot and makes
    /// the old node non-current; the slot count stays below the node count.
    #[test]
    fn rediscovery_shares_a_slot_and_supersedes_the_old_node() {
        let mut graph = SearchGraph::new(cfg(0));
        assert_eq!(graph.slot(graph.root()), 0);
        assert!(graph.is_current(graph.root()));
        assert_eq!(graph.num_configs(), 1);

        let (a, _) = graph.insert(graph.root(), MoveSet::new([lane(0, 0, 0)]), cfg(1), 5.0);
        assert_eq!(graph.slot(a), 1);
        assert!(graph.is_current(a));
        assert_eq!(graph.num_configs(), 2);

        let (b, is_new) = graph.insert(graph.root(), MoveSet::new([lane(0, 0, 1)]), cfg(1), 2.0);
        assert!(is_new);
        assert_ne!(a, b);
        assert_eq!(
            graph.slot(b),
            graph.slot(a),
            "same configuration, same slot"
        );
        assert!(graph.is_current(b));
        assert!(
            !graph.is_current(a),
            "the superseded node is no longer current"
        );
        assert_eq!(graph.seen_id(&cfg(1)), Some(b));
        assert_eq!(graph.num_configs(), 2);
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.len() - graph.num_configs(), 1, "one superseded id");

        // A costlier rediscovery changes nothing.
        let (c, is_new) = graph.insert(graph.root(), MoveSet::new([lane(0, 0, 0)]), cfg(1), 9.0);
        assert!(!is_new);
        assert_eq!(c, b);
        assert_eq!(graph.num_configs(), 2);
    }

    #[test]
    fn reconstruct_path_root_is_empty() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let graph = SearchGraph::new(root_cfg);
        let path = graph.reconstruct_path(graph.root());
        assert!(path.is_empty());
    }

    #[test]
    fn reconstruct_path_depth_3() {
        let cfg0 = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(cfg0);

        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let cfg1 = Config::new([(0, loc(0, 1))]).unwrap();
        let (id1, _) = graph.insert(graph.root(), ms1.clone(), cfg1, 1.0);

        let ms2 = MoveSet::new([lane(0, 1, 0)]);
        let cfg2 = Config::new([(0, loc(0, 2))]).unwrap();
        let (id2, _) = graph.insert(id1, ms2.clone(), cfg2, 2.0);

        let ms3 = MoveSet::new([lane(0, 2, 0)]);
        let cfg3 = Config::new([(0, loc(0, 3))]).unwrap();
        let (id3, _) = graph.insert(id2, ms3.clone(), cfg3, 3.0);

        let path = graph.reconstruct_path(id3);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], ms1);
        assert_eq!(path[1], ms2);
        assert_eq!(path[2], ms3);
    }

    #[test]
    fn moveset_canonical_ordering() {
        let a = MoveSet::new([lane(0, 1, 0), lane(0, 0, 0)]);
        let b = MoveSet::new([lane(0, 0, 0), lane(0, 1, 0)]);
        assert_eq!(a, b);
    }

    #[test]
    fn moveset_deduplicates() {
        let ms = MoveSet::new([lane(0, 0, 0), lane(0, 0, 0)]);
        assert_eq!(ms.len(), 1);
    }

    #[test]
    fn moveset_decode_roundtrip() {
        let original = vec![lane(0, 0, 0), lane(1, 2, 3)];
        let ms = MoveSet::new(original.clone());
        let decoded = ms.decode();
        // Decoded should contain same lanes (order may differ from input
        // but MoveSet sorts them).
        assert_eq!(decoded.len(), 2);
        assert!(decoded.contains(&original[0]));
        assert!(decoded.contains(&original[1]));
    }

    #[test]
    fn insert_same_config_equal_cost_rejected() {
        let root_cfg = Config::new([(0, loc(0, 0))]).unwrap();
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]).unwrap();
        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let (first_id, _) = graph.insert(graph.root(), ms1, child_cfg.clone(), 1.0);

        let ms2 = MoveSet::new([lane(0, 0, 1)]);
        let (returned_id, is_new) = graph.insert(graph.root(), ms2, child_cfg, 1.0);

        assert!(!is_new);
        assert_eq!(returned_id, first_id);
    }

    /// `NodeId`s are dense arena indices: every id equals the node's position,
    /// ids are handed out consecutively from 0, and none is ever reused.
    ///
    /// The entropy driver's per-node caches index `Vec`s by `NodeId` directly
    /// instead of hashing, which is only sound while this holds. The transposition
    /// path is the interesting case: a cheaper re-discovery *appends* a new node
    /// and repoints the table, leaving the old id valid and its slot occupied —
    /// lazy deletion, so indices are never invalidated or compacted.
    #[test]
    fn node_ids_are_dense_arena_indices() {
        let mut graph = SearchGraph::new(Config::new([(0, loc(0, 0))]).unwrap());
        assert_eq!(graph.root().0, 0);
        assert_eq!(graph.len(), 1);

        let a = graph
            .insert(graph.root(), MoveSet::new([lane(0, 0, 0)]), cfg(1), 1.0)
            .0;
        let b = graph
            .insert(a, MoveSet::new([lane(0, 0, 0)]), cfg(2), 2.0)
            .0;
        assert_eq!((a.0, b.0), (1, 2), "ids are consecutive from the root");
        assert_eq!(graph.len(), 3);

        // Re-discover `cfg(2)` more cheaply: a *new* id appears, the old one
        // stays addressable, and the arena only grows.
        let (cheaper, is_new) =
            graph.insert(graph.root(), MoveSet::new([lane(0, 0, 0)]), cfg(2), 0.5);
        assert!(is_new);
        assert_eq!(cheaper.0, 3, "re-discovery appends rather than reusing");
        assert_eq!(graph.len(), 4);
        assert_eq!(
            graph.g_score(b),
            2.0,
            "the superseded node is still readable"
        );

        // Every id ever handed out is a valid index into a `Vec` sized to `len()`.
        for id in [graph.root(), a, b, cheaper] {
            assert!((id.0 as usize) < graph.len());
        }
    }
}
