#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p data

# Test 1: 10 rows, 5 columns, skip 1 col
cat << 'EOF' > data/test_10.csv
0,1.0,2.0,3.0,4.0
0,1.1,2.1,3.1,4.1
0,1.2,2.2,3.2,4.2
0,1.3,2.3,3.3,4.3
0,1.4,2.4,3.4,4.4
0,1.5,2.5,3.5,4.5
0,1.6,2.6,3.6,4.6
0,1.7,2.7,3.7,4.7
0,1.8,2.8,3.8,4.8
0,1.9,2.9,3.9,4.9
EOF

./bin/csv_to_bin --in data/test_10.csv --out data/test_10.bin --skip-cols 1
SIZE1=$(stat -c%s data/test_10.bin)
echo "Test 1 size: ${SIZE1} (expected 176)"
if [ "${SIZE1}" -ne 176 ]; then
    echo "Test 1 FAILED"
    exit 1
fi

# Test 2: semicolon with ? row
cat << 'EOF' > data/test_power.csv
Date;Time;Global_active_power;Global_reactive_power;Voltage
16/12/2006;17:24:00;4.216;0.418;234.84
16/12/2006;17:25:00;?;?;?
16/12/2006;17:26:00;5.360;0.436;233.63
EOF

./bin/csv_to_bin --in data/test_power.csv --out data/test_power.bin --delim ';' --skip-cols 2 --skip-header --skip-missing
SIZE2=$(stat -c%s data/test_power.bin)
echo "Test 2 size: ${SIZE2} (expected 40)"
if [ "${SIZE2}" -ne 40 ]; then
    echo "Test 2 FAILED"
    exit 1
fi

rm -f data/test_10.csv data/test_10.bin data/test_power.csv data/test_power.bin
echo "All csv_to_bin unit tests PASSED!"
