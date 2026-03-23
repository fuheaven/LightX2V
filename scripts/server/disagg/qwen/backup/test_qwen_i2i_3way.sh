#!/bin/bash
# Test Qwen I2I 3-way disagg: check services then run post script.
# Usage: run after start_qwen_i2i_disagg_3way.sh is up.
#   cd /home/fuhaiwen/LightX2V && bash scripts/server/disagg/qwen/test_qwen_i2i_3way.sh
#
# Optional: set IMAGE_PATH for I2I input, or edit post_qwen_i2i_3way.py.

set -e

lightx2v_path=${LIGHTX2V_PATH:-/home/fuhaiwen/LightX2V}
ENCODER_URL=${ENCODER_URL:-http://localhost:8012}
TRANSFORMER_URL=${TRANSFORMER_URL:-http://localhost:8013}
DECODER_URL=${DECODER_URL:-http://localhost:8014}

echo "Checking Qwen I2I 3-way services..."
for url in "$ENCODER_URL" "$TRANSFORMER_URL" "$DECODER_URL"; do
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$url/v1/service/status" 2>/dev/null | grep -q 200; then
        echo "  OK $url"
    else
        echo "  FAIL $url (is the service running?)"
        exit 1
    fi
done

echo "Running post_qwen_i2i_3way.py ..."
cd "${lightx2v_path}"
python scripts/server/disagg/qwen/post_qwen_i2i_3way.py
echo "Done. Check save_results/qwen_i2i_disagg_3way.png on the Decoder node."
