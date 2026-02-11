use kiy_engine::search::{TTEntry, TTFlag, TranspositionTable};

#[test]
fn test_tt_new_zero_size_and_store_get() {
    let tt = TranspositionTable::new(0);
    let entry = TTEntry {
        hash: 0xdeadbeef,
        depth: 5,
        score: 42,
        flag: TTFlag::Exact,
        best_move: None,
        static_eval: 50,
    };
    tt.store(entry);
    let got = tt.get(entry.hash).expect("entry should be found");
    assert_eq!(got.hash, entry.hash);
    assert_eq!(got.depth, entry.depth);
    assert_eq!(got.score, entry.score);
    assert_eq!(got.static_eval, entry.static_eval);
}

#[test]
fn test_tt_static_eval_round_trip() {
    let tt = TranspositionTable::new(1);

    // Test positive, negative, and zero static evals
    for &eval in &[0i32, 100, -100, 500, -500, 32767, -32768] {
        let entry = TTEntry {
            hash: 0x12345678u64.wrapping_add(eval as u64),
            depth: 10,
            score: 42,
            flag: TTFlag::Exact,
            best_move: None,
            static_eval: eval,
        };
        tt.store(entry);
        let got = tt.get(entry.hash).expect("entry should be found");
        assert_eq!(got.static_eval, eval, "static_eval mismatch for {}", eval);
        assert_eq!(got.score, 42, "score corrupted for eval={}", eval);
        assert_eq!(got.depth, 10, "depth corrupted for eval={}", eval);
    }
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
        static_eval: 15,
    };
    tt.store(entry);
    assert!(tt.get(entry.hash).is_some());
    tt.clear();
    assert!(tt.get(entry.hash).is_none());
}

#[test]
fn test_tt_all_flags_with_static_eval() {
    let tt = TranspositionTable::new(1);
    for (i, &flag) in [TTFlag::Exact, TTFlag::LowerBound, TTFlag::UpperBound].iter().enumerate() {
        let entry = TTEntry {
            hash: 0xfeed0000 + i as u64,
            depth: 5,
            score: -200,
            flag,
            best_move: None,
            static_eval: -150,
        };
        tt.store(entry);
        let got = tt.get(entry.hash).expect("should be found");
        assert_eq!(got.flag, flag);
        assert_eq!(got.score, -200);
        assert_eq!(got.static_eval, -150);
    }
}
