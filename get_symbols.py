import pytz
from datetime import datetime, timedelta
from gym_mtsim import MtSimulator, OrderType, Timeframe

if __name__ == '__main__':
    FOREX_DATA_PATH = './'
    sim = MtSimulator(
        unit='USD',
        balance=10000.,
        leverage=100.,
        stop_out_level=0.2,
        hedge=False,
    )

    sim.download_data(
        symbols=['EURUSD', 'GBPCAD', 'GBPUSD', 'USDCAD', 'USDCHF', 'GBPJPY', 'USDJPY'],
        time_range=(
            datetime(2001, 5, 5, tzinfo=pytz.UTC),
            datetime(2024, 11, 5, tzinfo=pytz.UTC)
        ),
        timeframe=Timeframe.D1
    )
    sim.save_symbols(FOREX_DATA_PATH)
