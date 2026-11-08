#!/usr/bin/env bash
set -euo pipefail

SOLR_URL="${SOLR_URL:-http://192.168.1.100:8983/solr/noticias}"
CORE="${CORE:-noticias}"

docker --context server stack deploy -c docker-stack.yml taller-rag

for _ in $(seq 1 90); do
  if curl -sf -m 2 http://192.168.1.100:8983/solr/ >/dev/null; then break; fi
  sleep 2
done

CID=$(docker --context server ps -q --filter name=taller-rag_solr | head -1)
if [ -z "$CID" ]; then
  echo "no encontre la tarea de Solr en el stack"
  exit 1
fi

docker --context server exec "$CID" solr create -c "$CORE" -d _default || true

curl -sS -X POST -H 'Content-Type: application/json' --data-binary @- "$SOLR_URL/schema" <<'JSON'
{"add-field":[
  {"name":"chunk_id","type":"string","stored":true,"indexed":true},
  {"name":"text","type":"text_general","stored":true,"indexed":true},
  {"name":"document_id","type":"string","stored":true,"indexed":true},
  {"name":"title","type":"text_general","stored":true,"indexed":true},
  {"name":"url","type":"string","stored":true,"indexed":false},
  {"name":"collection_name","type":"string","stored":true,"indexed":true},
  {"name":"source","type":"string","stored":true,"indexed":true},
  {"name":"language","type":"string","stored":true,"indexed":true},
  {"name":"created_at","type":"string","stored":true,"indexed":false},
  {"name":"position","type":"pint","stored":true,"indexed":false},
  {"name":"end_offset","type":"pint","stored":true,"indexed":false},
  {"name":"chunk_strategy","type":"string","stored":true,"indexed":false},
  {"name":"text_sha256","type":"string","stored":true,"indexed":false}
]}
JSON

echo "Solr listo en $SOLR_URL"
