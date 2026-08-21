# Pacman Python Debugger

Cette webapp React/Vite rejoue une trace pédagogique du projet Python.

## Lancer

Depuis ce dossier :

```bash
npm install
npm run dev
```

Puis ouvrir l'URL indiquée par Vite.

Pour produire le build :

```bash
npm run build
```

## Utilisation

- Choisir `run.py` ou `train.py`.
- Utiliser les boutons précédent/suivant, lecture/pause et reset.
- Les flèches gauche/droite naviguent entre les étapes.
- La barre espace lance ou arrête la lecture.
- La colonne centrale affiche le code et la ligne courante.
- La colonne droite affiche la pile, les arguments, les variables et l'état visuel.

La simulation est déterministe et fonctionne sans backend Python. Elle expose les lignes et les valeurs représentatives de l'exécution actuelle du projet ; elle ne lance pas PyTorch dans le navigateur.

