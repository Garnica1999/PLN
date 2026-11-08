#!/usr/bin/env bash
set -euo pipefail

SOLR_URL="${SOLR_URL:-http://192.168.1.100:8983/solr/noticias}"
CORE="${CORE:-noticias}"

if ! docker --context server ps --format '{{.Names}}' | grep -qx taller-rag-solr; then
  docker --context server run -d --name taller-rag-solr --restart unless-stopped \
    -p 8983:8983 -v taller-rag_solr_data:/var/solr solr:9
fi

for _ in $(seq 1 60); do
  if curl -sf -m 2 http://192.168.1.100:8983/solr/ >/dev/null; then break; fi
  sleep 2
done

docker --context server exec taller-rag-solr solr create -c "$CORE" -d _default || true

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
