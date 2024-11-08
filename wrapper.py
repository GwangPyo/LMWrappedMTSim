from gym_mtsim.envs.mt_env import MtEnv, MtSimulator, OrderType
from gymnasium.spaces import Text
from gymnasium.spaces import Discrete
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime
from typing import Callable, Union, Tuple
import json


class LanguageMTSim(MtEnv):

    def __init__(
            self,
            original_simulator: MtSimulator,
            trading_symbols: List[str],
            window_size: int,
            time_points: Optional[List[datetime]] = None,
            fee: Union[float, Callable[[str], float]] = 0.0005,
            symbol_max_orders: int = 1,
            multiprocessing_processes: Optional[int] = None,
            render_mode: Optional[str] = None,
            preprocess: Optional[Callable] = np.arcsinh,
            randomize_initial_balance: bool = False,
            initial_balance_kwargs: Optional[Tuple[float, float]] = None,
            time_split: bool = False,
            min_time_split_length: int = 10,
            risk_free_rate: float = 0.02,
            max_time_limit: int = 200,
            risk_premium: bool = False,
            done_if_equity_zero: bool = False,
            loss_cut: Optional[float] = None,
            log_reward: bool = False,
            seed: int = 42,
    ) -> None:
        super().__init__(
            original_simulator=original_simulator,
            trading_symbols=trading_symbols,
            window_size=window_size,
            time_points=time_points,
            fee=fee,
            symbol_max_orders=symbol_max_orders,
            multiprocessing_processes=multiprocessing_processes,
            render_mode=render_mode,
            preprocess=preprocess,
            randomize_initial_balance=randomize_initial_balance,
            initial_balance_kwargs=initial_balance_kwargs,
            time_split=time_split,
            min_time_split_length=min_time_split_length,
            risk_free_rate=risk_free_rate,
            max_time_limit=max_time_limit,
            risk_premium=risk_premium,
            done_if_equity_zero=done_if_equity_zero,
            loss_cut=loss_cut,
            log_reward=log_reward,
            seed=seed,
        )
        self.original_observation_space = self.observation_space

        self.observation_space = Text(max_length=2000, min_length=10, charset=''.join(chr(i) for i in range(128)))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        return self.string_process(obs), info

    def step(self, action: np.ndarray) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        next_obs, reward, done, timeout, info = super().step(action)

        for symbol in self.trading_symbols:
            error_string = info['orders'][symbol]['error']

            if error_string != '':
                error_reason = error_string.split('(')[0]
                next_obs['symbols'][symbol] += f"Error: {error_reason}"

        return self.string_process(next_obs), reward, done, timeout, info

    def string_process(self, obs_string: Dict[str, str]):
        account_info = f"Account Info {obs_string['account_info']}\n"
        symbol_string = ""
        for s in self.trading_symbols:
            symbol_string += f"{s}: {obs_string['symbols'][s]} \n"
        output = account_info + symbol_string
        return output

    def _apply_action(self, action: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        orders_info = {}
        closed_orders_info = {symbol: [] for symbol in self.trading_symbols}

        k = self.symbol_max_orders + 2
        for i, symbol in enumerate(self.trading_symbols):
            symbol_action = action[k * i:k * (i + 1)]
            close_orders_probability = symbol_action[:-2]
            hold_probability = symbol_action[-2]
            volume = symbol_action[-1] * 100
            hold = self.np_rng.choice([False, True], p=[1 - hold_probability, hold_probability])
            modified_volume = self._get_modified_volume(symbol, volume)
            symbol_orders = self.simulator.symbol_orders(symbol)
            if len(symbol_orders) > 0:
                prob = close_orders_probability[:len(symbol_orders)]
                closes = np.asarray([self.np_rng.choice([False, True], p=[1 - p, p]) for p in prob])
                orders_to_close_index = np.where(closes)[0]
                orders_to_close = np.array(symbol_orders)[orders_to_close_index]

            else:
                orders_to_close = []

            for j, order in enumerate(orders_to_close):
                self.simulator.close_order(order)
                closed_orders_info[symbol].append(dict(
                    order_id=order.id, symbol=order.symbol, order_type=order.type,
                    volume=order.volume, fee=order.fee,
                    margin=order.margin, profit=order.profit,
                    close_probability=close_orders_probability[orders_to_close_index][j],
                ))

            orders_capacity = self.symbol_max_orders - (len(symbol_orders) - len(orders_to_close))
            orders_info[symbol] = dict(
                order_id=None, symbol=symbol, hold_probability=hold_probability,
                hold=hold, volume=volume, capacity=orders_capacity, order_type=None,
                modified_volume=modified_volume, fee=float('nan'), margin=float('nan'),
                error='',
            )

            if self.simulator.hedge and orders_capacity == 0:
                orders_info[symbol].update(dict(
                    error="cannot add more orders"
                ))
            elif not hold:
                order_type = OrderType.Buy if volume > 0. else OrderType.Sell
                fee = self.fee if type(self.fee) is float else self.fee(symbol)

                try:
                    order = self.simulator.create_order(order_type, symbol, modified_volume, fee)
                    new_info = dict(
                        order_id=order.id, order_type=order_type,
                        fee=fee, margin=order.margin,
                    )
                except ValueError as e:
                    new_info = dict(error=str(e))

                orders_info[symbol].update(new_info)

        return orders_info, closed_orders_info

    def _get_observation(self) -> Dict[str, str]:
        keys = self.signal_features.keys()
        features = {k: self.signal_features[k][(self._current_tick - self.window_size + 1):(self._current_tick + 1)]
                    for k in keys}
        features = self.extract_features(features)
        orders = np.zeros(self.original_observation_space['orders'].shape)
        for i, symbol in enumerate(self.trading_symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            symbol_order_strings = ""
            for j, order in enumerate(symbol_orders):

                orders[i, j] = [order.entry_price, order.volume, order.profit]
                if order.type == -1:
                    order_type = "Sell"
                else:
                    order_type = "Buy"

                order_string = (f"OrderID: {order.id}, OrderType:{order_type} "
                                f"Entry Price: {order.entry_price:.2f} "
                                f"Order Margin: {order.margin:.2f} Order Profit: {order.profit:.2f}")
                symbol_order_strings += order_string

            features[symbol] = features[symbol] + symbol_order_strings

        account_info = (f"Balance: {self.simulator.balance:.2f} Equity: {self.simulator.equity:.2f} "
                        f"Margin: {self.simulator.margin:.2f}: Leverage: {int(self.simulator.leverage)}"
                        f"Symbol Max Order: {self.symbol_max_orders}")
        observation = {"account_info": account_info,
                       "symbols": features
                       }

        return observation

    def extract_features(self, features: Dict[str, np.ndarray]):
        keys = features.keys()
        extracted_features = {}
        for k in keys:
            target_price = features[k]
            ma_5 = self.moving_average(target_price, 5)
            ma_10 = self.moving_average(target_price, 10)
            rsi_14 = self.rsi(target_price, 14)
            william_r_14 = self.calculate_last_williams_r(target_price, 14)
            low = self.low(target_price)
            high = self.high(target_price)
            open_price = target_price[-1, 1]
            string_information = (f"Open: {open_price: .2f} "
                                  f"MA5: {ma_5: .2f} "
                                  f"MA10: {ma_10: .2f} "
                                  f"RSI(14): {rsi_14: .2f} "
                                  f"William_R(14): {william_r_14: .2f} "
                                  f"Low: {low: .2f} ",
                                  f"High: {high: .2f} ",
                                  )
            extracted_features[k] = "".join(string_information)
        return extracted_features

    def _process_data(self) -> Dict:
        class CDict(dict):
            @property
            def shape(self, ):
                return np.column_stack(list(self.values())).shape

        signal_features = CDict(**self.prices)
        return signal_features

    def _get_prices(self, keys: List[str] = ['Close', 'Open', 'High', 'Low', 'Volume']) -> Dict[str, np.ndarray]:
        prices = {}
        for symbol in self.trading_symbols:
            get_price_at = lambda time: \
                self.original_simulator.price_at(symbol, time)[keys]
            if self.multiprocessing_pool is None:
                p = list(map(get_price_at, self.time_points))
            else:
                p = self.multiprocessing_pool.map(get_price_at, self.time_points)

            prices[symbol] = np.array(p)
        return prices

    @staticmethod
    def moving_average(prices: np.ndarray, days: int = 5):
        close = prices[:, 0]
        return close[days:].mean()

    @staticmethod
    def rsi(prices: np.ndarray, days: int = 14):
        close_prices = prices[:, 0]
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-days:])
        avg_loss = np.mean(losses[-days:])

        if avg_loss == 0:
            last_rsi = 100
        else:
            rs = avg_gain / avg_loss
            last_rsi = 100 - (100 / (1 + rs))
        return last_rsi

    @staticmethod
    def calculate_last_williams_r(prices: np.ndarray, days: int = 14):
        # data: numpy array of shape [DAY, 5] where columns are ['Close', 'Open', 'High', 'Low', 'Volume']
        close_prices = prices[:, 0]
        high_prices = prices[:, 2]
        low_prices = prices[:, 3]
        highest_high = np.max(high_prices[-days:])
        lowest_low = np.min(low_prices[-days:])
        last_close = close_prices[-1]
        williams_r = -100 * (highest_high - last_close) / (highest_high - lowest_low)
        return williams_r

    @staticmethod
    def low(prices):
        return np.min(prices[:, -2])

    @staticmethod
    def high(prices):
        return np.max(prices[:, -3])

    @staticmethod
    def average_volume(prices):
        return np.mean(prices[:, -1])

    @property
    def string_action_template(self):
        types = ["Sell", "Buy", "Hold"]
        index = 0
        string_result = {}
        for s in self.trading_symbols:
            orders = [str("{" + str(f'"Action Type": "{types[index]}", "Volume": 0.05') + "}")]
            for i in range(self.symbol_max_orders - 1):
                orders.append('{"Action Type": "Hold", "Volume": 0.0}')
            index += 1
            string_result[s] = str(orders)
        string_result = ",".join(f'"{k}": {v}' for k, v in string_result.items())
        string_result = str(string_result).replace('\\', '')
        string_result = str(string_result).replace("'", "")
        string_result = '{' + string_result + '}'

        return str(string_result)

    def parse_action(self, string_action):
        dict_actions: Dict[str, list] = json.loads(string_action)
        array_actions = []
        for s in self.trading_symbols:
            target = dict_actions[s]
            close_probs = []
            hold_probs = []
            volumes = []
            for i in range(len(target)):
                order_1 = target[i]
                order_type = order_1['Action Type'].lower()
                order_volume = order_1['Volume']
                close_prob = 0
                hold_prob = 0
                sign = 0
                if order_type != 'hold':
                    close_prob = 1
                    if order_type == 'sell':
                        sign = -1
                    else:
                        sign = 1
                else:
                    hold_prob = 1
                volume = sign * order_volume
                close_probs.append(close_prob)
                hold_probs.append(hold_prob)
                volumes.append(volume)
            hold_prob = np.mean(hold_probs).item()
            volume = np.mean(volumes).item()
            actions = close_probs + [hold_prob, volume]
            actions = np.asarray(actions)
            array_actions.append(actions)
        arr_act = np.concatenate(array_actions, axis=-1)
        return arr_act

    def base_llm_message(self):
        role_describer = "You are an expert of forex trading given trading information, decide whether "
        "buy, sell, hold and its volume for each symbol."
        formatting_guide_line = ("system will raise exception if you don't follow the format. "
                                 "Please keep the example template strictly.")
        example_output = str(f'Example Output: {self.string_action_template}')
        message = [{"role": "system", "content": role_describer},
                   {"role": "system", "content": formatting_guide_line},
                   {"role": "system", "content": example_output}
                   ]
        return message

    def exception_llm_message(self):
        messages = [
            {"role": "system",
             "content": "You are an expert of forex trading given trading information, decide whether "
                        "buy, sell, hold and its volume for each symbol."
                        "You will minimize the loss risk of conditional value at risk 10%."},
            {"role": "system", "content": f'Example Output: {self.string_action_template}'},
            {"role": "system", "content": 'System raised exception because you did not follow the example format.'
                                          'Please follow the example format'}
        ]
        return messages

    def message_template(self, obs: str):
        return self.base_llm_message() + [{"role": "user", "content": obs}]

    def exception_message_template(self, obs: str):
        return self.exception_llm_message() + [{"role": "user", "content": obs}]