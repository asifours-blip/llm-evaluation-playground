"""``serve`` binds a real loopback socket; verify it answers and shuts down cleanly.

No external network access: everything here talks to 127.0.0.1 on an
OS-assigned ephemeral port that this test starts and always tears down.
"""

from __future__ import annotations

import threading
import urllib.request

from rag_quality_lab.web.app import serve


def test_serve_answers_and_shuts_down_cleanly() -> None:
    server = serve(host="127.0.0.1", port=0, allow_live=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as response:
            assert response.status == 200
            assert b"RAG Quality Lab" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
