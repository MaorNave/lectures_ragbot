"""The chat page and its widget must never mix versions: a new index.html with a cached old widget.js breaks the page."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from smartup_rag.api.app import create_app
from smartup_rag.config import Settings


@pytest.fixture
def client():
    return TestClient(create_app(Settings(_env_file=None, warm_on_startup=False, keep_warm_minutes=0)))


@pytest.mark.parametrize("path", ["/widget/widget.js", "/widget/widget.css", "/widget/fonts/rubik-latin-wght.woff2"])
def test_widget_files_are_revalidated_on_every_load(client, path):
    r = client.get(path)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert r.headers.get("etag")  # so revalidating an unchanged file is a cheap 304


def test_the_page_asks_for_the_exact_widget_version(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    js = re.search(r'src="/widget/widget\.js\?v=([0-9a-f]{10})"', r.text)
    css = re.search(r'href="/widget/widget\.css\?v=([0-9a-f]{10})"', r.text)
    assert js and css and js.group(1) == css.group(1)
    assert client.get(f"/widget/widget.js?v={js.group(1)}").status_code == 200


def test_versioned_widget_files_are_cached_for_good(client):
    r = client.get("/widget/widget.js?v=0123456789")
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]


def test_an_unchanged_page_is_a_304(client):
    first = client.get("/")
    again = client.get("/", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304 and not again.content
