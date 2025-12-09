"""Tests for history fetcher with caching and DataFrame support."""
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock
from osrs_flipper.history import HistoryFetcher


@pytest.fixture
def mock_client():
    """Mock OSRS client for testing."""
    client = Mock()
    return client


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    return cache


@pytest.fixture
def sample_api_data():
    """Sample data from API (using API field names)."""
    return [
        {
            "timestamp": 1000,
            "avgHighPrice": 100,
            "avgLowPrice": 90,
            "highPriceVolume": 50,
            "lowPriceVolume": 30,
        },
        {
            "timestamp": 2000,
            "avgHighPrice": 110,
            "avgLowPrice": 95,
            "highPriceVolume": 60,
            "lowPriceVolume": 40,
        },
    ]


class TestHistoryFetcher:
    """Test suite for HistoryFetcher class."""

    def test_fetches_and_caches_new_item(self, mock_client, mock_cache, sample_api_data):
        """Test fetching new item with no cache."""
        # Arrange
        mock_cache.get.return_value = None
        mock_client.fetch_timeseries.return_value = sample_api_data

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        result = fetcher.get_history(item_id=4151, timestep="1h")

        # Assert
        mock_cache.get.assert_called_once_with(4151, "1h")
        mock_client.fetch_timeseries.assert_called_once_with(
            item_id=4151, timestep="1h", timestamp=None
        )
        mock_cache.store.assert_called_once_with(4151, "1h", sample_api_data)
        assert result == sample_api_data

    def test_uses_cache_and_fetches_incremental(
        self, mock_client, mock_cache, sample_api_data
    ):
        """Test incremental fetch when cache exists."""
        # Arrange
        cached_data = [sample_api_data[0]]
        new_data = [sample_api_data[1]]

        mock_cache.get.return_value = cached_data
        mock_cache.get_latest_timestamp.return_value = 1000
        mock_client.fetch_timeseries.return_value = new_data

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        result = fetcher.get_history(item_id=4151, timestep="1h")

        # Assert
        mock_cache.get.assert_called_once_with(4151, "1h")
        mock_cache.get_latest_timestamp.assert_called_once_with(4151, "1h")
        mock_client.fetch_timeseries.assert_called_once_with(
            item_id=4151, timestep="1h", timestamp=1000
        )
        mock_cache.append.assert_called_once_with(4151, "1h", new_data)
        assert result == cached_data

    def test_force_refresh_ignores_cache(
        self, mock_client, mock_cache, sample_api_data
    ):
        """Test force_refresh parameter bypasses cache."""
        # Arrange
        mock_cache.get.return_value = [{"old": "data"}]
        mock_client.fetch_timeseries.return_value = sample_api_data

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        result = fetcher.get_history(item_id=4151, timestep="1h", force_refresh=True)

        # Assert
        # Should NOT call cache.get since force_refresh=True
        mock_cache.get.assert_not_called()
        mock_client.fetch_timeseries.assert_called_once_with(
            item_id=4151, timestep="1h", timestamp=None
        )
        mock_cache.store.assert_called_once_with(4151, "1h", sample_api_data)
        assert result == sample_api_data

    def test_get_dataframe(self, mock_client, mock_cache, sample_api_data):
        """Test DataFrame conversion with derived columns."""
        # Arrange
        mock_cache.get.return_value = sample_api_data
        mock_cache.get_latest_timestamp.return_value = 2000
        mock_client.fetch_timeseries.return_value = []

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        df = fetcher.get_dataframe(item_id=4151, timestep="1h")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        # Check columns
        expected_columns = [
            "timestamp",
            "high",
            "low",
            "mid_price",
            "high_volume",
            "low_volume",
            "total_volume",
            "buyer_ratio",
        ]
        assert list(df.columns) == expected_columns

        # Check first row values
        row = df.iloc[0]
        assert row["timestamp"] == 1000
        assert row["high"] == 100
        assert row["low"] == 90
        assert row["mid_price"] == 95  # (100 + 90) // 2
        assert row["high_volume"] == 50
        assert row["low_volume"] == 30
        assert row["total_volume"] == 80  # 50 + 30
        assert abs(row["buyer_ratio"] - 0.625) < 0.001  # 50 / 80

    def test_get_dataframe_handles_zero_volume(
        self, mock_client, mock_cache
    ):
        """Test DataFrame handles zero volume edge case."""
        # Arrange
        zero_vol_data = [
            {
                "timestamp": 1000,
                "avgHighPrice": 100,
                "avgLowPrice": 90,
                "highPriceVolume": 0,
                "lowPriceVolume": 0,
            }
        ]
        mock_cache.get.return_value = zero_vol_data
        mock_cache.get_latest_timestamp.return_value = 1000
        mock_client.fetch_timeseries.return_value = []

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        df = fetcher.get_dataframe(item_id=4151, timestep="1h")

        # Assert
        assert len(df) == 1
        assert df.iloc[0]["total_volume"] == 0
        assert df.iloc[0]["buyer_ratio"] == 0.0  # Should handle division by zero

    def test_creates_default_client_and_cache(self):
        """Test that default client and cache are created if None."""
        # Act
        fetcher = HistoryFetcher()

        # Assert
        assert fetcher.client is not None
        assert fetcher.cache is not None
        assert hasattr(fetcher.client, 'fetch_timeseries')
        assert hasattr(fetcher.cache, 'get')

    def test_get_dataframe_empty_data(self, mock_client, mock_cache):
        """Test DataFrame with empty data."""
        # Arrange
        mock_cache.get.return_value = []
        mock_cache.get_latest_timestamp.return_value = None
        mock_client.fetch_timeseries.return_value = []

        fetcher = HistoryFetcher(client=mock_client, cache=mock_cache)

        # Act
        df = fetcher.get_dataframe(item_id=4151, timestep="1h")

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        expected_columns = [
            "timestamp",
            "high",
            "low",
            "mid_price",
            "high_volume",
            "low_volume",
            "total_volume",
            "buyer_ratio",
        ]
        assert list(df.columns) == expected_columns
