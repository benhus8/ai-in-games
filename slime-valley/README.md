# Slime Valley

## Uruchomienie

```bash
uv sync
uv run python main.py
uv run python main.py --agent heuristic
```

## Sterowanie

- `W`: skok
- `A`: ruch w lewo
- `D`: ruch w prawo
- `Q`: wyjscie z gry

Mozesz wciskac kilka klawiszy jednoczesnie (np. `W` + `A` albo `W` + `D`).

Po uruchomieniu sterujesz prawym slime'em. Lewy slime jest botem. Jesli nic nie wciskasz, Twoj slime stoi w miejscu.

## Agent heurystyczny

W folderze `agents` jest prosty bot oparty o heurystyke. Przewiduje punkt spadku pilki na podstawie aktualnej predkosci i ustawia slime'a w tym miejscu, a skok wykonuje dopiero przy pilce opadajacej nad jego strona.

Uruchomienie:

```bash
uv run python main.py --agent heuristic
```
