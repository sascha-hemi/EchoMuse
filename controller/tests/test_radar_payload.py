"""A Biscuit ARM payload must never be used for a Radar installation."""
import json

import pytest
import em_api
import em_emos_build as eb


@pytest.mark.asyncio
async def test_radar_refuses_release_with_only_biscuit_inits(monkeypatch):
    bundle = eb.build_payload_bundle({'init': b'aarch64', 'init32': b'biscuit'}, 'test')
    async def release():
        return {'version': 'test', 'assets': {em_api.EMOS_PAYLOAD_ASSET: {'url': 'test'}}}
    async def binary(*args):
        return bundle
    monkeypatch.setattr(em_api, '_fetch_latest_emos_release', release)
    monkeypatch.setattr(em_api, '_fetch_binary', binary)
    init, sbin, version, error = await em_api._fetch_emos_payload('arm', 'radar')
    assert init is None and sbin == {}
    assert error.status == 404
    assert 'init32-radar' in error.text
