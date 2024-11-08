from wrapper import LanguageMTSim, MtSimulator
import openai
import json


if __name__ == '__main__':
    KEY = 'KEY HERE' # Recommend path enrolled open-ai key for safety issue.
    # This is just example for the direct
    simulator = MtSimulator(unit='USD',
                            balance=10000.,
                            leverage=100.,
                            stop_out_level=0.2,
                            hedge=False,
                            symbols_filename='./data/symbols_forex_train.pkl',
                            )
    symbols = ['USDCAD', 'USDJPY']
    env = LanguageMTSim(original_simulator=simulator, trading_symbols=symbols, window_size=30,
                        seed=64
                        )

    client = openai.OpenAI(api_key=KEY)
    obs, _ = env.reset()
    exception_counter = 0
    for _ in range(10):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=env.message_template(obs)
            )
            output = str(completion.choices[0].message.content)
            action = env.parse_action(output)
        except json.decoder.JSONDecodeError:
            exception_counter += 1
            print(f"exception {exception_counter}")
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=env.exception_message_template(obs)
            )
            output = str(completion.choices[0].message.content)
            action = env.parse_action(output)

        finally:
            obs, reward, done, timeout, info = env.step(action)
            print(reward)
