"""
Reinforcement Learning: Estrategia de trading
"""
# %%
# Imports
from mlfin.trading.trader import Trader
from mlfin.trading.functions import get_state


# %%
# Obtengo datos y configuro el problema
with open('data/spx.csv') as csvfile:
    lines = csvfile.read().splitlines()
    prices = [float(line.split(",")[1]) for line in lines[1:]]

data = prices[-100:]  # Tomo últimos 100 días

window_size = 10  # La regla de trading contempla últimos 10 días
repeticiones = 2  # Voy a recorrer 2 veces los datos
batch_size = 20  # Cantidad de jornadas que el trader utilizará para revisar su regla.


# %%
# Entreno al trader
trader = Trader(window_size)
n = len(data) - 1
sells = (0, 0)

for e in range(repeticiones):
    print(56 * '=')
    print(f'Corrida {e + 1} de {repeticiones}'.center(53))
    print(56 * '-')
    state = get_state(data, 0, window_size + 1)

    total_profit = 0.
    trader.book = []

    for t in range(n):
        print(f'Day {t:3} | ', end='')

        # Trader decide
        action, has_conv = trader.act(state)

        # Hold
        if action == 0:
            reward = 0

            print(f'     Hold      |                  |',
                  f'{"Convicción" if has_conv else "Azar"}'.center(10))
        # Buy
        elif action == 1:
            reward = 0
            trader.book.append(data[t])

            print(f'Buy:  {data[t]:8.2f} |                  |',
                  f'{"Convicción" if has_conv else "Azar"}'.center(10))

        # Sell
        elif action == 2:
            bought_price = trader.book.pop(0)  # FIFO
            profit = data[t] - bought_price
            reward = max(profit, 0)

            if has_conv:
                sells = (sells[0] + 1, sells[1] + 1 if profit > 0 else sells[1])

            total_profit += profit
            print(f'Sell: {data[t]:8.2f} | Profit: {profit:8.2f} |',
                  f'{"Convicción" if has_conv else "Azar"}'.center(10))

        done = True if t == n - 1 else False
        next_state = get_state(data, t + 1, window_size + 1)
        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            total_profit += data[t] * len(trader.book) - sum(trader.book)
            print(56 * '-')
            print(f'Total Profit:    {total_profit:8.2f}')
            if sells[0] > 0:
                print(f'Sells Hit Ratio: {sells[1] / sells[0]:8.2%}')
            print(56 * '=', '\n')

        # Si ya tiene una mínima experiencia que revise su regla.
        if len(trader.memory) > batch_size:
            trader.learn(batch_size)
