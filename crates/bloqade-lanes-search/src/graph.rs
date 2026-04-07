//! Arena-based search graph with transposition table.
//!
//! [`SearchGraph`] stores search nodes in a flat arena (`Vec<NodeData>`)
//! indexed by [`NodeId`]. A transposition table maps configurations to the
//! best-known node (lowest g-score), using the actual cost rather than depth.
//!
//! Path reconstruction walks parent pointers — no children are stored.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::config::Config;

/// Opaque handle to a node in the search graph.
///
/// Internally an index into the node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

/// A set of lanes applied simultaneously in one move step.
///
/// Stored as a sorted, deduplicated `Vec<u64>` of
/// [`LaneAddr::encode_u64()`] values, making it order-independent
/// (analogous to Python's `frozenset[LaneAddress]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MoveSet {
    lanes: Vec<u64>,
}

impl MoveSet {
    /// Create from an iterator of lane addresses.
    ///
    /// The lanes are encoded, sorted, and deduplicated.
    pub fn new(lanes: impl IntoIterator<Item = LaneAddr>) -> Self {
        let mut encoded: Vec<u64> = lanes.into_iter().map(|l| l.encode_u64()).collect();
        encoded.sort_unstable();
        encoded.dedup();
        Self { lanes: encoded }
    }

    /// Create from pre-encoded lane u64 values. Sorts and deduplicates.
    pub fn from_encoded(mut encoded: Vec<u64>) -> Self {
        encoded.sort_unstable();
        encoded.dedup();
        Self { lanes: encoded }
    }

    /// Decode back to `LaneAddr` values.
    pub fn decode(&self) -> Vec<LaneAddr> {
        self.lanes
            .iter()
            .map(|&bits| LaneAddr::decode_u64(bits))
            .collect()
    }

    /// Number of lanes in this move set.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Returns `true` if the move set contains no lanes.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }
}

/// Internal node storage.
struct NodeData {
    config: Config,
    parent: Option<NodeId>,
    parent_move: Option<MoveSet>,
    g_score: f64,
}

/// Arena-based search graph with transposition table.
///
/// Nodes are stored in a flat `Vec` and referenced by [`NodeId`].
/// The transposition table maps each unique configuration to the
/// [`NodeId`] with the lowest known g-score.
///
/// Unlike the Python `ConfigurationTree`:
/// - Uses g-score (cost) for the transposition table, not depth.
/// - Does not store children — only parent pointers for path reconstruction.
/// - Arena allocation avoids reference cycles and per-node heap allocation.
pub struct SearchGraph {
    nodes: Vec<NodeData>,
    seen: HashMap<Config, NodeId>,
}

impl std::fmt::Debug for SearchGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchGraph")
            .field("num_nodes", &self.nodes.len())
            .field("num_seen", &self.seen.len())
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
        };
        let mut seen = HashMap::new();
        seen.insert(root, NodeId(0));
        Self {
            nodes: vec![root_node],
            seen,
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

    /// Get the depth of a node (number of steps from root).
    /// Walks parent pointers, so O(depth).
    pub fn depth(&self, id: NodeId) -> u32 {
        let mut d: u32 = 0;
        let mut current = id;
        while let Some(parent_id) = self.nodes[current.0 as usize].parent {
            d += 1;
            current = parent_id;
        }
        d
    }

    /// Number of nodes in the arena (always >= 1 due to root).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Always returns `false` — the graph always contains at least the root.
    ///
    /// Provided to satisfy the `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Look up the best-known [`NodeId`] for a configuration.
    pub fn seen_id(&self, config: &Config) -> Option<NodeId> {
        self.seen.get(config).copied()
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
        if let Some(&existing_id) = self.seen.get(&new_config) {
            let existing_g = self.nodes[existing_id.0 as usize].g_score;
            if existing_g <= new_g {
                // Already seen at equal-or-lower cost.
                return (existing_id, false);
            }
            // Re-discovered at lower cost: create new node, update table.
        }

        let new_id = NodeId(self.nodes.len() as u32);
        self.nodes.push(NodeData {
            config: new_config.clone(),
            parent: Some(parent),
            parent_move: Some(move_set),
            g_score: new_g,
        });
        self.seen.insert(new_config, new_id);
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
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, LocationAddr, MoveType};

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            word_id: word,
            site_id: site,
        }
    }

    fn lane(word: u32, site: u32, bus: u32) -> LaneAddr {
        LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            word_id: word,
            site_id: site,
            bus_id: bus,
        }
    }

    #[test]
    fn root_creation() {
        let cfg = Config::new([(0, loc(0, 0))]);
        let graph = SearchGraph::new(cfg.clone());

        assert_eq!(graph.len(), 1);
        assert_eq!(*graph.config(graph.root()), cfg);
        assert_eq!(graph.g_score(graph.root()), 0.0);
    }

    #[test]
    fn insert_new_config() {
        let root_cfg = Config::new([(0, loc(0, 0))]);
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]);
        let ms = MoveSet::new([lane(0, 0, 0)]);
        let (id, is_new) = graph.insert(graph.root(), ms, child_cfg, 1.0);

        assert!(is_new);
        assert_eq!(graph.g_score(id), 1.0);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn insert_same_config_higher_cost_rejected() {
        let root_cfg = Config::new([(0, loc(0, 0))]);
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]);
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
        let root_cfg = Config::new([(0, loc(0, 0))]);
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]);
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

    #[test]
    fn reconstruct_path_root_is_empty() {
        let root_cfg = Config::new([(0, loc(0, 0))]);
        let graph = SearchGraph::new(root_cfg);
        let path = graph.reconstruct_path(graph.root());
        assert!(path.is_empty());
    }

    #[test]
    fn reconstruct_path_depth_3() {
        let cfg0 = Config::new([(0, loc(0, 0))]);
        let mut graph = SearchGraph::new(cfg0);

        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let cfg1 = Config::new([(0, loc(0, 1))]);
        let (id1, _) = graph.insert(graph.root(), ms1.clone(), cfg1, 1.0);

        let ms2 = MoveSet::new([lane(0, 1, 0)]);
        let cfg2 = Config::new([(0, loc(0, 2))]);
        let (id2, _) = graph.insert(id1, ms2.clone(), cfg2, 2.0);

        let ms3 = MoveSet::new([lane(0, 2, 0)]);
        let cfg3 = Config::new([(0, loc(0, 3))]);
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
        let root_cfg = Config::new([(0, loc(0, 0))]);
        let mut graph = SearchGraph::new(root_cfg);

        let child_cfg = Config::new([(0, loc(0, 1))]);
        let ms1 = MoveSet::new([lane(0, 0, 0)]);
        let (first_id, _) = graph.insert(graph.root(), ms1, child_cfg.clone(), 1.0);

        let ms2 = MoveSet::new([lane(0, 0, 1)]);
        let (returned_id, is_new) = graph.insert(graph.root(), ms2, child_cfg, 1.0);

        assert!(!is_new);
        assert_eq!(returned_id, first_id);
    }
}
