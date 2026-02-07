// src/search/mcts_memory.rs
//! Memory-efficient storage for MCTS nodes
//!
//! Prevents memory exhaustion during long searches by:
//! - Arena allocation for nodes
//! - LRU cache for less-frequented branches
//! - Automatic garbage collection of old subtrees

use crate::search::mcts::Node;
use chess::Board;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Arena-based node allocator for memory efficiency
pub struct NodeArena {
    /// Storage for nodes (indexed by generation -> index)
    generations: Mutex<Vec<Vec<Option<NodeEntry>>>>,
    /// Current generation number
    current_gen: AtomicUsize,
    /// Free list for recycling
    free_list: Mutex<Vec<NodeRef>>,
    /// Total allocated nodes
    total_allocated: AtomicUsize,
    /// Maximum nodes per generation before GC
    max_per_generation: usize,
}

/// A node entry in the arena
#[allow(dead_code)]
struct NodeEntry {
    node: Arc<Node>,
    last_accessed: Instant,
    access_count: AtomicUsize,
}

/// Reference to a node in the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeRef {
    pub generation: usize,
    pub index: usize,
}

impl NodeArena {
    /// Create a new node arena
    pub fn new(max_per_generation: usize) -> Self {
        Self {
            generations: Mutex::new(vec![Vec::with_capacity(1000)]),
            current_gen: AtomicUsize::new(0),
            free_list: Mutex::new(Vec::new()),
            total_allocated: AtomicUsize::new(0),
            max_per_generation,
        }
    }

    /// Allocate a new node in the arena
    pub fn allocate(&self, board: Board) -> (NodeRef, Arc<Node>) {
        // Try to recycle from free list
        if let Some(node_ref) = self.free_list.lock().pop() {
            let node = Arc::new(Node::new(board));

            // Update the entry
            let mut gens = self.generations.lock();
            if let Some(gen) = gens.get_mut(node_ref.generation) {
                if let Some(entry) = gen.get_mut(node_ref.index) {
                    *entry = Some(NodeEntry {
                        node: node.clone(),
                        last_accessed: Instant::now(),
                        access_count: AtomicUsize::new(0),
                    });
                    return (node_ref, node);
                }
            }
        }

        // Allocate new slot
        let node = Arc::new(Node::new(board));
        let mut gens = self.generations.lock();
        let current_gen = self.current_gen.load(Ordering::Relaxed);

        // Check if current generation is full
        if gens[current_gen].len() >= self.max_per_generation {
            // Create new generation
            gens.push(Vec::with_capacity(1000));
            self.current_gen.fetch_add(1, Ordering::Relaxed);
        }

        let gen_idx = self.current_gen.load(Ordering::Relaxed);
        let node_idx = gens[gen_idx].len();

        gens[gen_idx].push(Some(NodeEntry {
            node: node.clone(),
            last_accessed: Instant::now(),
            access_count: AtomicUsize::new(0),
        }));

        self.total_allocated.fetch_add(1, Ordering::Relaxed);

        (
            NodeRef {
                generation: gen_idx,
                index: node_idx,
            },
            node,
        )
    }

    /// Get node by reference
    pub fn get(&self, node_ref: NodeRef) -> Option<Arc<Node>> {
        let gens = self.generations.lock();
        gens.get(node_ref.generation)
            .and_then(|gen| gen.get(node_ref.index))
            .and_then(|entry| {
                if let Some(e) = entry {
                    e.access_count.fetch_add(1, Ordering::Relaxed);
                    Some(e.node.clone())
                } else {
                    None
                }
            })
    }

    /// Total allocated nodes
    pub fn total_nodes(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Run garbage collection on old generations
    pub fn gc(&self, keep_generations: usize) -> usize {
        let mut gens = self.generations.lock();
        let current = self.current_gen.load(Ordering::Relaxed);

        let mut freed = 0;
        for gen_idx in 0..current.saturating_sub(keep_generations) {
            if let Some(gen) = gens.get_mut(gen_idx) {
                for (idx, entry) in gen.iter_mut().enumerate() {
                    if entry.is_some() {
                        // Add to free list
                        self.free_list.lock().push(NodeRef {
                            generation: gen_idx,
                            index: idx,
                        });
                        *entry = None;
                        freed += 1;
                    }
                }
            }
        }

        freed
    }
}

/// LRU cache for MCTS subtrees
pub struct MCTSLRUCache {
    /// Maximum cache size
    max_size: usize,
    /// Cached nodes indexed by board hash
    cache: RwLock<HashMap<u64, CachedSubtree>>,
    /// LRU order (most recent at back)
    lru_order: Mutex<VecDeque<u64>>,
}

/// A cached MCTS subtree
#[allow(dead_code)]
struct CachedSubtree {
    root: Arc<Node>,
    depth: usize,
    visits: usize,
    last_accessed: Instant,
}

impl MCTSLRUCache {
    /// Create new LRU cache
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: RwLock::new(HashMap::new()),
            lru_order: Mutex::new(VecDeque::new()),
        }
    }

    /// Lookup a cached subtree by board hash
    pub fn lookup(&self, board_hash: u64) -> Option<Arc<Node>> {
        let cache = self.cache.read();

        if let Some(entry) = cache.get(&board_hash) {
            // Update LRU order
            let mut order = self.lru_order.lock();
            order.retain(|&h| h != board_hash);
            order.push_back(board_hash);

            return Some(entry.root.clone());
        }

        None
    }

    /// Insert a subtree into cache
    pub fn insert(&self, board_hash: u64, root: Arc<Node>, depth: usize, visits: usize) {
        // Check if we need to evict
        {
            let cache = self.cache.read();
            let order = self.lru_order.lock();

            if cache.len() >= self.max_size && !cache.contains_key(&board_hash) {
                // Evict oldest
                drop(cache);
                drop(order);
                self.evict_oldest();
            }
        }

        // Insert new entry
        let mut cache = self.cache.write();
        let mut order = self.lru_order.lock();

        cache.insert(
            board_hash,
            CachedSubtree {
                root,
                depth,
                visits,
                last_accessed: Instant::now(),
            },
        );

        order.retain(|&h| h != board_hash);
        order.push_back(board_hash);
    }

    /// Evict oldest entry from cache
    fn evict_oldest(&self) {
        let mut order = self.lru_order.lock();
        let mut cache = self.cache.write();

        if let Some(oldest) = order.pop_front() {
            cache.remove(&oldest);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read();
        let _order = self.lru_order.lock();

        CacheStats {
            size: cache.len(),
            max_size: self.max_size,
            total_visits: cache.values().map(|e| e.visits).sum(),
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        let mut order = self.lru_order.lock();
        cache.clear();
        order.clear();
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub total_visits: usize,
}

/// Tree manager combining arena and cache
pub struct MCTSTreeManager {
    arena: NodeArena,
    cache: MCTSLRUCache,
    max_memory_mb: usize,
}

impl MCTSTreeManager {
    /// Create new tree manager
    pub fn new(max_memory_mb: usize, cache_size: usize) -> Self {
        let arena_size = max_memory_mb * 1024 * 1024 / std::mem::size_of::<NodeEntry>() / 2;

        Self {
            arena: NodeArena::new(arena_size),
            cache: MCTSLRUCache::new(cache_size),
            max_memory_mb,
        }
    }

    /// Create root node
    pub fn create_root(&self, board: Board) -> (NodeRef, Arc<Node>) {
        self.arena.allocate(board)
    }

    /// Get node by reference
    pub fn get_node(&self, node_ref: NodeRef) -> Option<Arc<Node>> {
        self.arena.get(node_ref)
    }

    /// Lookup cached subtree
    pub fn lookup_cache(&self, board: &Board) -> Option<Arc<Node>> {
        self.cache.lookup(board.get_hash())
    }

    /// Cache a subtree
    pub fn cache_subtree(&self, board: &Board, root: Arc<Node>, depth: usize, visits: usize) {
        self.cache.insert(board.get_hash(), root, depth, visits);
    }

    /// Memory usage estimate in MB
    pub fn memory_usage_mb(&self) -> usize {
        let node_count = self.arena.total_nodes();
        let node_size = std::mem::size_of::<Node>() + std::mem::size_of::<NodeEntry>();
        let cache_size = self.cache.stats().size * std::mem::size_of::<CachedSubtree>();

        (node_count * node_size + cache_size) / (1024 * 1024)
    }

    /// Check if memory limit is exceeded
    pub fn is_memory_limited(&self) -> bool {
        self.memory_usage_mb() >= self.max_memory_mb
    }

    /// Run garbage collection
    pub fn gc(&self) -> usize {
        self.arena.gc(3) // Keep last 3 generations
    }

    /// Clear all storage
    pub fn clear(&self) {
        self.gc();
        self.cache.clear();
    }
}
