# Evil Chess Engine

## Overview
A Python chess engine with that also includes a minimal CLI to let a human play against it.

This is not a fully polished UCI engine. This is an entirely experimental project.

## Requirements
```python-chess``` 

```cython-chess``` **__(you will also need to have a working C++ compiler on your computer)__**

For more info, visit: [cython-chess](https://github.com/Ranuja01/cython-chess)

## Files

```main.py```, ```zobrist.py```, ```transposition.py``` - the core chess engine and its components. Implements tapered piece squared tables in evaluation, king safety heuristics, move ordering, alpha–beta pruning, quiescence search, transposition tables, LMR, iterative deepening, among other optimizations.

```simple_game_manager.py``` - A minimal command-line interface that lets a human player play against the engine by providing moves in **UCI notation.**

## How to play against the engine

### 1. Run the game manager (```simple_game_manager.py```)

`python simple_game_manager.py`

### 2. Initialize a board

The program will prompt you: `do you want to initialize the board state from a predetermined FEN? (y) yes (n) no`

Press `y` to set a custom FEN.

Invalid FENs will be rejected and you will be prompted again.

Press `n`  (or any other string) to use the default chess starting position.


### 3. Pick your color

The program will prompt you `do you want to play as white? (y) yes (n) no`

Press `y` to play as white.

Press `n` (or any other string) to play as black.

### 4. Enter moves

After you pick your board state and your color, the game will start. You must submit your moves in UCI formatting `ex: e2e4` for the engine to be able to read it.

### 5. Engine thinking time

By default, the engine's approximate thinking time is set to in `simple_game_manager.py` by this call at the end of the script:

`play_game(10)`

You can adjust the engine's thinking time to make it stronger or weaker. **This time limit is not a hard cap, but rather a guideline. Actual thinking time will vary between 6-20 seconds.** 

### 6. Game End

Once a terminal posiiton is reached, the program will stop and the result will be printed.



