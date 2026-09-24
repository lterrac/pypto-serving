#!/bin/bash
# Serve Qwen3-14B from the C++ binary over HTTP and exercise it with curl.
set -uo pipefail

export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib:/mounted_home/code/tokenizers-cpp-arm/install/lib:${LD_LIBRARY_PATH:-}

ROOT=/mounted_home/.cpp-port/pypto-serving
PORT=8771
cd "$ROOT" || exit 1

echo "[http] phys=${TASK_PHYS_DEVICE:-?} logical=${TASK_DEVICE:-?}"

./serving/build-full/pypto-serving-cpp \
  --model /mounted_home/models/Qwen3-14B \
  --device "${TASK_DEVICE:-0}" \
  --platform a2a3 \
  --max-model-len 512 \
  --host 127.0.0.1 --port "$PORT" \
  --repo-root "$ROOT" > /tmp/serving_http.log 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[http] stopping server (pid $SERVER_PID)"
  kill -TERM "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait for readiness. The model takes ~80s to come up, so poll /health rather
# than guessing; a 503 means "up but not ready", which is the distinction the
# C++ /health exists to make.
echo "[http] waiting for the engine ..."
READY=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[http] server died during startup"; tail -30 /tmp/serving_http.log; exit 1
  fi
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=1; echo "[http] ready after ${i}s"; break; fi
  sleep 1
done
[ "$READY" = "1" ] || { echo "[http] never became ready"; tail -30 /tmp/serving_http.log; exit 1; }

echo
echo "=============== GET /health ==============="
curl -sS -i "http://127.0.0.1:$PORT/health" | head -1
curl -sS "http://127.0.0.1:$PORT/health"; echo

echo
echo "=============== GET /v1/models ==============="
curl -sS "http://127.0.0.1:$PORT/v1/models"; echo

echo
echo "=============== POST /v1/completions ==============="
curl -sS "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}'; echo

echo
echo "=============== POST /v1/completions (stream) ==============="
curl -sS -N "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8, "stream": true}'

echo
echo "=============== POST /v1/chat/completions ==============="
curl -sS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Name one colour. Reply with just the word."}], "max_tokens": 12}'; echo

echo
echo "=============== POST /v1/chat/completions (stream) ==============="
curl -sS -N "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 6, "stream": true}'

echo
echo "=============== error handling ==============="
echo -n "malformed JSON  -> "; curl -sS -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d '{not json'
echo -n "missing prompt  -> "; curl -sS -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d '{"max_tokens": 4}'
# Distinct tokens: a run of repeated characters merges down under BPE and would
# be admitted, which is not what this is testing.
LONG_PROMPT=$(seq 1 2000 | tr '\n' ' ')
echo -n "prompt too long -> "; curl -sS -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' -d "{\"prompt\": \"$LONG_PROMPT\"}"

echo
echo "=============== two requests at once ==============="
curl -sS "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 5}' > /tmp/a.json &
PID_A=$!
curl -sS "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of Italy is", "max_tokens": 5}' > /tmp/b.json &
PID_B=$!
# Wait for these two only. A bare `wait` also waits for the server, which never
# exits, and the script hangs long after the requests have succeeded.
wait "$PID_A" "$PID_B"
echo "A: $(cat /tmp/a.json)"
echo "B: $(cat /tmp/b.json)"

echo
echo "[http] server log tail:"
tail -5 /tmp/serving_http.log | grep -vE "STRACE|mono_ns" | head -5
echo "[http] done"
