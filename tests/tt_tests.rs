use kiy_engine_v5_alpha::search::{TTEntry, TTFlag, TranspositionTable};

#[test]
fn test_tt_new_zero_size_and_store_get() {
    let tt = TranspositionTable::new(0);
    let entry = TTEntry {
        hash: 0xdeadbeef,
        depth: 5,
        score: 42,
        flag: TTFlag::Exact,
        best_move: None,
    };
    tt.store(entry);
    let got = tt.get(entry.hash).expect("entry should be found");
    assert_eq!(got.hash, entry.hash);
    assert_eq!(got.depth, entry.depth);
    assert_eq!(got.score, entry.score);
}

#[test]
fn test_tt_clear() {
    let tt = TranspositionTable::new(1);
    let entry = TTEntry {
        hash: 0xabc,
        depth: 2,
        score: 10,
        flag: TTFlag::LowerBound,
        best_move: None,
    };
    tt.store(entry);
    assert!(tt.get(entry.hash).is_some());
    tt.clear();
    assert!(tt.get(entry.hash).is_none());
}
