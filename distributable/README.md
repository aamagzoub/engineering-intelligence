# Sudanese Wist — Play Against AI

A standalone card game where you play Sudanese Wist against 3 AI opponents.

## How to Run (from source)

```
cd ..
python distributable/play_wist.py
```

## How to Build .exe (for distribution)

1. Run `build.bat` (installs PyInstaller and builds)
2. Find `dist/SudaneseWist.exe`
3. Share the .exe — no Python needed on the target machine

## How to Play

1. Click **Start Game**
2. See your 13 cards at the bottom
3. When it's your turn to bid: select trump suit → confirm or pass
4. When it's your turn to play: click a highlighted (green border) card
5. Win more tricks than the other team!

## Rules

- 4 players, 2 teams (you + Player 1 vs Player 2 + Player 4)
- Must follow suit if able
- Trump suit beats everything
- First card of the game must be trump
- Team that wins more tricks than their bid scores points
