"""Tests for historical data caching."""
import os
import tempfile
import pytest


class TestHistoricalCache:
    def test_cache_stores_data(self):
        from osrs_flipper.cache import HistoricalCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HistoricalCache(cache_dir=tmpdir)
            data = [
                {"timestamp": 1000, "avgHighPrice": 100, "avgLowPrice": 95},
                {"timestamp": 2000, "avgHighPrice": 105, "avgLowPrice": 100},
            ]
            cache.store(item_id=4151, timestep="1h", data=data)
            assert os.path.exists(os.path.join(tmpdir, "4151_1h.json"))

    def test_cache_retrieves_data(self):
        from osrs_flipper.cache import HistoricalCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HistoricalCache(cache_dir=tmpdir)
            data = [{"timestamp": 1000, "avgHighPrice": 100}]
            cache.store(item_id=4151, timestep="1h", data=data)
            retrieved = cache.get(item_id=4151, timestep="1h")
            assert retrieved == data

    def test_cache_returns_none_for_missing(self):
        from osrs_flipper.cache import HistoricalCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HistoricalCache(cache_dir=tmpdir)
            assert cache.get(item_id=99999, timestep="1h") is None

    def test_cache_get_latest_timestamp(self):
        from osrs_flipper.cache import HistoricalCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HistoricalCache(cache_dir=tmpdir)
            data = [
                {"timestamp": 1000}, {"timestamp": 3000}, {"timestamp": 2000}
            ]
            cache.store(item_id=4151, timestep="1h", data=data)
            assert cache.get_latest_timestamp(item_id=4151, timestep="1h") == 3000

    def test_cache_append_new_data(self):
        from osrs_flipper.cache import HistoricalCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HistoricalCache(cache_dir=tmpdir)
            data1 = [{"timestamp": 1000}, {"timestamp": 2000}]
            cache.store(item_id=4151, timestep="1h", data=data1)
            data2 = [{"timestamp": 2000}, {"timestamp": 3000}]  # 2000 is duplicate
            cache.append(item_id=4151, timestep="1h", new_data=data2)
            retrieved = cache.get(item_id=4151, timestep="1h")
            timestamps = [d["timestamp"] for d in retrieved]
            assert len(timestamps) == 3
            assert sorted(timestamps) == [1000, 2000, 3000]
