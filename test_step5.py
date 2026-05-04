import urllib.request
import urllib.error

req = urllib.request.Request('http://localhost:8000/v1/sessions', data=b'{"user_id": "alice", "plan_tier": "pro"}', headers={'Content-Type': 'application/json'}, method='POST')
try:
    print('PRO:', urllib.request.urlopen(req).read().decode())
except Exception as e:
    print('PRO ERR:', e)

req2 = urllib.request.Request('http://localhost:8000/v1/sessions', data=b'{"user_id": "alice", "plan_tier": "gold"}', headers={'Content-Type': 'application/json'}, method='POST')
try:
    print('GOLD:', urllib.request.urlopen(req2).read().decode())
except urllib.error.HTTPError as e:
    print('GOLD HTTP Error:', e.code)
