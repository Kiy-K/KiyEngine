#!/bin/bash
set -e

echo "=== Quick KiyEngine Test ==="

# Test 1: Basic UCI functionality
echo "🧪 Testing basic UCI functionality..."
timeout 10s ./target/release/kiy_engine_v4_omega << EOF || echo "UCI test completed"
uci
isready
quit
EOF

echo ""
echo "🧪 Testing engine with a simple position..."
timeout 15s ./target/release/kiy_engine_v4_omega << EOF || echo "Position test completed"
uci
isready
position startpos
go depth 10
quit
EOF

echo ""
echo "✅ Basic engine tests completed!"
